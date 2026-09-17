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
# Prefixes a UI session's cache identity. Kept short because it lands in dict keys.
UI_CACHE_IDENTITY_PREFIX = "ui:"


class AmbientAuthorizedToken(str):
    """A UI session's own saved token: entitled to ambient, so never behind the probe.

    Only ``allow_ambient_token`` tells it from an API key's token. ``str`` subclass so every
    consumer sees the string it is; sending your own credential must not buy less than none.
    """

    __slots__ = ()


def hf_token_arg(hf_token: Optional[str], *, allow_ambient_token: bool) -> HfTokenArg:
    """Return the explicit token, or choose ambient versus anonymous access."""
    token = (hf_token or "").strip()
    if token:
        return AmbientAuthorizedToken(token) if allow_ambient_token else token
    return None if allow_ambient_token else False


# Mirrors the list hub/services/download_lifecycle.py scrubs for download workers.
# HF_OIDC_RESOURCE names a token rather than holding one: hub >= 1.23 exchanges it inside
# get_token(), ahead of HF_TOKEN, so a scrubbed child would still resolve the operator's.
_HF_TOKEN_ENV_KEYS = (
    "HF_TOKEN",
    "HF_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HF_OIDC_RESOURCE",
)

# auth_check takes no timeout, nor the session under it (0.x: none; 1.30: Timeout(None)).
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
    trimmed = (hf_token or "").strip() or None
    # str.strip returns a plain str, which would demote a UI session to an API key.
    if trimmed is not None and isinstance(hf_token, AmbientAuthorizedToken):
        return AmbientAuthorizedToken(trimmed)
    return trimmed


def is_anonymous(hf_token: HfTokenArg) -> bool:
    """Named because a bare ``is False`` invites a ``not hf_token`` "simplification"."""
    return hf_token is False


def qualify_cache_identity(hf_token: HfTokenArg, digest: str) -> str:
    """Tag a token digest with the caller class that produced it.

    Same token value, different authorization: a UI session is entitled to ambient, an
    sk-unsloth API key is not. On the bare digest they collide, so a cache hands one the
    other's verdict and a coalescer merges their scans into whichever arrived first.
    """
    return (
        f"{UI_CACHE_IDENTITY_PREFIX}{digest}"
        if isinstance(hf_token, AmbientAuthorizedToken)
        else digest
    )


# Both signs: a revoked token must not keep reading, a flapping Hub must not be re-dialled.
_REPO_ACCESS_TTL_S = 60.0
# Says nothing about the credential. MUST exceed the probe timeout or every caller re-stalls.
_REPO_ACCESS_UNREACHABLE_TTL_S = 30.0
_REPO_ACCESS_CACHE_MAX = 1024
_repo_access_cache: dict[tuple[str, str, str], tuple[float, bool]] = {}
_repo_access_lock = threading.Lock()
# One probe per key: the probe runs outside _repo_access_lock, so a cold key would
# otherwise open a connection per caller.
_repo_access_inflight: dict[tuple[str, str, str], threading.Lock] = {}


class _ProbeTimedOut(Exception):
    """Raised when /auth-check could not be asked at all, rather than answering."""


# By name, so neither client is imported. A refusal or dead proxy is "could not ask" too.
_UNREACHABLE_EXC_NAMES = frozenset(
    {
        "Timeout",
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectError",
        "ConnectionError",
        "ProxyError",
        "NetworkError",
        "TransportError",
        "NameResolutionError",
    }
)
_UNREACHABLE_PACKAGES = frozenset({"requests", "httpx", "urllib3"})


def _is_probe_timeout(exc: BaseException) -> bool:
    """Could not ask, as opposed to asked and told no. Named for its original narrow case."""
    for cls in type(exc).__mro__:
        name = cls.__name__
        # Builtin ConnectionError is a real transport failure; looser names need a package.
        if name in {"Timeout", "ReadTimeout", "ConnectTimeout", "ConnectionError"}:
            return True
        module = getattr(cls, "__module__", "") or ""
        # Top-level package, not "httpx.": that prefix matched none of httpx's own.
        if module.split(".", 1)[0] in _UNREACHABLE_PACKAGES and (
            "Timeout" in name or name in _UNREACHABLE_EXC_NAMES
        ):
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
    offline: bool = False,
) -> bool:
    """Whether this caller may read the host Hub disk cache for *repo_id*.

    ``is_anonymous`` authenticates the caller CLASS, not the credential, so any token-shaped
    string leaves the sentinel and takes the disk fast paths. ``repo_info`` cannot replace
    the probe: gated public metadata still returns for an invalid token.

    ``True`` does not mean the token is valid. /auth-check answers "is this repo reachable",
    which a public repo answers 200 for any string; it discriminates on private and gated
    repos, which is where the cached reads are.

    Offline, an explicit token is denied without a memoized probe (fail closed) while ambient
    ``None`` still reads. ``offline`` is the CALLER's own flag: without it a cache-only
    request put its token on the wire and could stall for the probe timeout first.
    """
    if is_anonymous(hf_token):
        return False
    # Deny by default: `not isinstance(str)` gave the ambient answer to True, 1, b"...".
    if hf_token is None:
        return True
    if not isinstance(hf_token, str):
        return False
    if not hf_token:
        return True
    if isinstance(hf_token, AmbientAuthorizedToken):
        # Same caller class reads the cache tokenless from the None branch above.
        return True
    repo = (repo_id or "").strip()
    if not repo:
        return False
    if _is_local_path(repo):
        # Unvalidated in the URL, so probing puts the caller's path on the wire with their
        # bearer token, for an answer that can only be no.
        return False
    return _explicit_token_reaches_repo(repo, hf_token, repo_type, offline = offline)


def public_cache_read_authorized(
    *,
    repo_id: str,
    repo_type: str = "model",
    offline: bool = False,
) -> bool:
    """Whether serving *repo_id* from the cache to a caller with NO credential leaks anything.

    The sentinel can never authorize itself, which is right for a private repo and wrong for
    a public one. An unauthenticated /auth-check is exactly that difference. Fail closed when
    it cannot be asked.
    """
    repo = (repo_id or "").strip()
    if not repo or _is_local_path(repo):
        return False
    return _explicit_token_reaches_repo(repo, None, repo_type, offline = offline)


def cached_read_refused(
    hf_token: HfTokenArg,
    *,
    repo_id: str,
    is_cached,
    repo_type: str = "model",
    offline: bool = False,
) -> bool:
    """Refuse a read only where the operator's disk could answer it AND this caller may not.

    An uncached repo has nothing to leak: refusing it protects nothing and costs a legitimate
    caller its answer whenever the probe is merely unavailable rather than negative (a mirror
    without the undocumented /auth-check, one transient failure). ``is_cached`` is asked FIRST
    so nothing on disk means no probe, and must fail closed or the guard's own failure opens
    the path it guards. Each reader passes its own predicate: a file at a revision, a dataset
    in either cache, a repo dir.

    "May not read it" is not "cannot authorize itself", which is where the sentinel sits: a
    public repo is one it was always entitled to read. Asked here so every reader shares the
    rule; the dataset preview used to be the only one asking.
    """
    if not is_cached():
        return False
    if cache_reads_authorized(hf_token, repo_id = repo_id, repo_type = repo_type, offline = offline):
        return False
    return not (
        is_anonymous(hf_token)
        and public_cache_read_authorized(repo_id = repo_id, repo_type = repo_type, offline = offline)
    )


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
        # Fail open on the offline question only; authorization still needs a live probe.
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


def _explicit_token_reaches_repo(
    repo_id: str,
    token: Optional[str],
    repo_type: str,
    offline: bool = False,
) -> bool:
    # None asks the public question, under its own key: a public repo answers 200 for every
    # token, so a shared key would let any string claim that verdict.
    key = (
        repo_id.casefold(),
        repo_type,
        hashlib.sha256(token.encode()).hexdigest()[:16] if token else "anonymous",
    )
    cached = _cached_repo_access(key, time.monotonic())
    if cached is not None:
        return cached
    if offline or _hub_offline():
        return False

    with _inflight_lock(key):
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
        # AFTER the probe: `start + TTL` memoizes an expired entry when the Hub stalls.
        finished = time.monotonic()
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
        _repo_access_cache.clear()


def _inflight_lock(key: tuple[str, str, str]) -> threading.Lock:
    with _repo_access_lock:
        lock = _repo_access_inflight.get(key)
        if lock is None:
            if len(_repo_access_inflight) >= _REPO_ACCESS_CACHE_MAX:
                _repo_access_inflight.clear()
            lock = threading.Lock()
            _repo_access_inflight[key] = lock
        return lock


def _probe_endpoint() -> str:
    """The endpoint the rest of the backend means, not the raw env value.

    ``HfApi().endpoint`` returns ``HF_ENDPOINT`` verbatim, so a scheme-less mirror builds a
    URL both clients reject and every probe on that machine is denied. ``hf_endpoint_url``
    is where the backend already normalises it; falls back since this module sits beneath.
    """
    try:
        from utils.utils import hf_endpoint_url
        return hf_endpoint_url().rstrip("/")
    except Exception:
        from huggingface_hub import HfApi
        return HfApi().endpoint


def _same_probe_target(answered: str, asked: str) -> bool:
    """Whether two URLs address the same endpoint, up to the spellings a client may change.

    Case in the host and a default port carry no meaning; the path, query and everything
    else do, so a redirect to ``?next=login`` on the same path is still a different target.
    """
    from urllib.parse import urlsplit

    def _parts(url: str):
        parsed = urlsplit(url)
        scheme = (parsed.scheme or "").lower()
        port = parsed.port or {"http": 80, "https": 443}.get(scheme)
        return (
            scheme,
            (parsed.hostname or "").lower(),
            port,
            parsed.path.rstrip("/"),
            parsed.query,
        )

    try:
        return _parts(answered) == _parts(asked)
    except ValueError:
        # An unparseable final URL is not proof that the right repo answered.
        return False


def _probe_repo_access(repo_id: str, token: Optional[str], repo_type: str) -> bool:
    try:
        from huggingface_hub import constants
        from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status

        if repo_type not in constants.REPO_TYPES:
            return False
        # A raw "?" or "#" ends the path early at /api/{type}s/{id}, which answers 200 with
        # public metadata for a gated repo.
        from urllib.parse import quote

        # quote keeps "/", so ".." survives and dot-segment removal probes a different repo
        # than the one memoized.
        if any(segment in {".", ".."} for segment in repo_id.split("/")):
            return False
        path = f"{_probe_endpoint()}/api/{repo_type}s/{quote(repo_id, safe = '/')}/auth-check"
        response = get_session().get(
            path,
            # False, not None: None falls back to the ambient login, asking the public
            # question with the operator's own credential.
            headers = build_hf_headers(token = token if token else False),
            timeout = _REPO_ACCESS_PROBE_TIMEOUT_S,
        )
        hf_raise_for_status(response)
        # hf_raise_for_status passes 3xx: without this a bare 307 reads as authorized.
        if 300 <= getattr(response, "status_code", 0) < 400:
            return False
        # And get_session DOES follow them (httpx.Client(follow_redirects=True)), so the
        # check above never fires: a redirect to a login page returns an approving 200 for
        # somewhere else. Only the repo we asked about may answer for it.
        if getattr(response, "history", None):
            return False
        # Belt and braces for a client that does not record history. Compared on parts, not
        # as strings: httpx canonicalises response.url (lower-cases the host, drops an
        # explicit :443) while the configured HF_ENDPOINT keeps its spelling, so a raw
        # comparison denied every mirror written as HF-MIRROR.example or with the port.
        final_url = getattr(response, "url", None)
        if final_url is not None and not _same_probe_target(str(final_url), path):
            return False
        return True
    except Exception as exc:
        if _is_probe_timeout(exc):
            raise _ProbeTimedOut from exc
        return False
