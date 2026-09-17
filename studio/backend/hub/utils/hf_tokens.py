# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Request-scoped Hugging Face token helpers."""

from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from typing import Literal, MutableMapping, Optional, Union

logger = logging.getLogger(__name__)

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
# ``None`` is a THIRD value here, not a miss: "the Hub could not be asked". It is memoized so
# a dead endpoint is not re-dialled per call, and memoized as unknown rather than as False so
# the 30 s window resolves locally on every call instead of holding a denial the Hub never
# gave -- a download finishing inside the window takes effect at once, and a recovered Hub is
# re-probed when it expires. ``_CACHE_MISS`` is what absence looks like.
_repo_access_cache: dict[tuple[str, str, str], tuple[float, Optional[bool]]] = {}
_CACHE_MISS = object()
_repo_access_lock = threading.Lock()
# One probe per key: the probe runs outside _repo_access_lock, so a cold key would
# otherwise open a connection per caller.
_repo_access_inflight: dict[tuple[str, str, str], threading.Lock] = {}

# An answered NO outlives the verdict cache, and only an unaskable Hub ever reads it.
#
# Unaskable resolves against the disk, which is right for an offline operator and wrong as a
# way to overturn a refusal that has already been given: 429 and 5xx are unaskable, they are
# reachable from outside (a burst of probes rate-limits this host's own endpoint), and the
# 60 s verdict cache means a caller the Hub refused a minute ago is one outage away from the
# operator's cached copy of the repo. So a denial is remembered for longer than it is cached,
# and while it is remembered "could not ask" answers no rather than reading the disk.
#
# Not a lockout: a Hub that answers again replaces it (an answered True is memoized as usual
# and the memory is dropped), and a repo that was never refused has nothing here, so an
# air-gapped host, a Hub outage and a mirror without /auth-check are unaffected. Keyed exactly
# like the verdict cache, so it says nothing about any other credential.
#
# It has NO time limit, on purpose. A clock-based expiry hands the denial back on the one input
# the caller controls: wait it out, ask again while the Hub is unaskable, and _resolve_unaskable
# authorizes locally on the strength of the operator's disk -- for a credential the Hub refused,
# with nothing having changed but the time. Only an affirmative answer is evidence that the
# refusal no longer holds, so only an affirmative answer clears it. The size cap below is what
# bounds the table, and the entries are recorded in insertion order so the cap evicts the
# oldest refusal rather than an arbitrary one.
_denied_repo_access: dict[tuple[str, str, str], float] = {}


def _remember_denial(key: tuple[str, str, str], now: float) -> None:
    with _repo_access_lock:
        # Re-insert at the end so a refusal that is still being re-asked is not the first to be
        # evicted. The stored time is the eviction order only; it never expires the entry.
        _denied_repo_access.pop(key, None)
        while len(_denied_repo_access) >= _REPO_ACCESS_CACHE_MAX:
            _denied_repo_access.pop(next(iter(_denied_repo_access)), None)
        _denied_repo_access[key] = now


def _forget_denial(key: tuple[str, str, str]) -> None:
    with _repo_access_lock:
        _denied_repo_access.pop(key, None)


def _denial_is_remembered(key: tuple[str, str, str]) -> bool:
    with _repo_access_lock:
        return key in _denied_repo_access


def _with_remembered_denial(key: tuple[str, str, str], verdict: Optional[bool]) -> Optional[bool]:
    """A verdict of "could not ask" reads as the last answer the Hub gave, if it was no."""
    if verdict is None and _denial_is_remembered(key):
        return False
    return verdict


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


# An ANSWER of no, as opposed to a failure to get one. Hub raises these by class; the names
# are matched rather than imported so a version that drops or moves one cannot turn a denial
# into "could not ask". DisabledRepoError is a deliberate refusal too.
_DENIAL_EXC_NAMES = frozenset(
    {
        "RepositoryNotFoundError",
        "GatedRepoError",
        "DisabledRepoError",
        "RevisionNotFoundError",
    }
)
# 401/403 are the credential being rejected. 410 is a repo withdrawn, 451 one blocked: both
# answered. Every other status -- 404 with no HF error code, 405, 429, 5xx -- is the endpoint
# failing to answer the question we asked.
#
# Measured against the live endpoint rather than assumed, because the whole split rests on it:
#   public model or dataset            -> 200
#   gated model, no credential         -> 401  X-Error-Code: GatedRepo
#   private or nonexistent repo        -> 401  (no error code)
#   /auth-check on a path that is not a route -> 404, generic HTML, no error code
# So huggingface.co never answers 404 ABOUT A REPO: a bare 404 means this endpoint has no
# /auth-check, which is the HF_ENDPOINT mirror case, and is not a statement about the caller.
_DENIAL_STATUSES = frozenset({401, 403, 410, 451})


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
    """Drop memoized Hub access answers. Tests only.

    The remembered denials go with them: they outlive the verdict cache on purpose, so a test
    that only cleared the cache would carry one test's refusal into the next one's unaskable
    probe.
    """
    with _repo_access_lock:
        _repo_access_cache.clear()
        _repo_access_inflight.clear()
        _denied_repo_access.clear()


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

    Three outcomes, not two. A Hub that ANSWERED no still denies, which is the whole of the
    boundary this gate exists for. A Hub that could not be ASKED -- ``HF_HUB_OFFLINE``, the
    caller's own cache-only ``offline``, a connection refused, a timeout, a 5xx, a 429, or an
    ``HF_ENDPOINT`` mirror with no /auth-check route -- is not an answer, and turning it into
    a denial made a purely LOCAL question ("is this repo on my disk") depend on reaching
    huggingface.co: an operator offline with a fully downloaded model was told it was
    unavailable. Unaskable therefore falls back to the local fact, on disk or not, which is
    the question actually being asked; see ``_resolve_unaskable``.
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
    verdict = _explicit_token_reaches_repo(repo, hf_token, repo_type, offline = offline)
    if verdict is None:
        return _resolve_unaskable(repo, repo_type, token = hf_token)
    return verdict


def public_cache_read_authorized(
    *,
    repo_id: str,
    repo_type: str = "model",
    offline: bool = False,
) -> bool:
    """Whether serving *repo_id* from the cache to a caller with NO credential leaks anything.

    The sentinel can never authorize itself, which is right for a private repo and wrong for
    a public one. An unauthenticated /auth-check is exactly that difference.

    Failing closed when it cannot be asked is what made every anonymous caller on an offline
    host lose its own downloaded public models, so the unaskable case resolves against the
    disk here too, on the same rule as ``cache_reads_authorized``.
    """
    repo = (repo_id or "").strip()
    if not repo or _is_local_path(repo):
        return False
    verdict = _explicit_token_reaches_repo(repo, None, repo_type, offline = offline)
    if verdict is None:
        return _resolve_unaskable(repo, repo_type, token = None)
    return verdict


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

    Both questions below resolve an unaskable Hub against the disk rather than denying, so a
    reader behind this gate keeps working offline. Only a Hub that answered no refuses.
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


def _ambient_hf_token() -> "tuple[bool, Optional[str]]":
    """``(known, token)``: the credential THIS HOST downloads with.

    ``huggingface_hub.get_token()`` is the authority. It is what every download in this
    process resolves, so it covers the env aliases, the OIDC exchange and the token file
    together, and asking it is the only way to be sure the answer matches what actually
    populated the cache.

    THREE outcomes, not two, and the third is the one that matters. ``(True, "hf_...")`` the
    host has this credential; ``(True, None)`` the Hub library says the host has none;
    ``(False, None)`` the question could not be answered here. Collapsing the third into
    "this host has no credential" is a fail-open: the token file could hold one this process
    could not read, and a credential-less caller would then be told the cache contains
    nothing it lacks permission for. Unknown therefore authorizes nobody.

    The env keys are the fallback for a hub too old to export ``get_token``, and finding one
    there IS an answer; finding none is not, because the token file is the source that
    fallback cannot see.

    Not memoized. It is read only inside the unaskable branch, an operator can revoke or set
    a token at any moment, and a memo would decide authorization from a credential the host
    no longer has.
    """
    get_token = None
    try:
        from huggingface_hub import get_token as _get_token
        get_token = _get_token
    except Exception:
        get_token = None
    if get_token is not None:
        try:
            token = get_token()
        except Exception:
            # Asked and not answered. Not the same as "there is none".
            return (False, None)
        if isinstance(token, str) and token.strip():
            return (True, token.strip())
        return (True, None)
    import os

    for key in _HF_TOKEN_ENV_KEYS:
        if key == "HF_OIDC_RESOURCE":
            # Names a token rather than holding one, so it cannot be compared to a caller's.
            continue
        value = os.environ.get(key)
        if isinstance(value, str) and value.strip():
            return (True, value.strip())
    return (False, None)


def _saved_studio_hf_token() -> "tuple[bool, Optional[str]]":
    """``(known, token)``: the HF credential the Studio UI saves and downloads with.

    Separate from ``_ambient_hf_token`` because it is a separate store and neither can see
    the other. Settings writes this one into the encrypted credential store
    (``credential_secrets.save_hf_token``); nothing writes it to the token file or to
    ``HF_TOKEN``, so ``huggingface_hub.get_token()`` does not return it, and the UI hands it
    back on each request in ``X-Unsloth-HF-Token`` rather than relying on ambient pickup.

    It is therefore just as much "the credential this host's cache was filled with" as the
    ambient one, and on the ordinary Studio install it is the ONLY one. Same three outcomes
    and the same reason for the third: an unreadable store (locked database, missing
    encryption key) is not "the host has no credential", and collapsing it into one would
    hand a credential-less caller a cache that may hold private bytes.

    Not memoized, for the reason ``_ambient_hf_token`` is not: the operator can save, rotate
    or clear this token at any moment from Settings.
    """
    try:
        from storage import credential_secrets
    except Exception:
        # The store exists in every Studio install; failing to import it is an unanswered
        # question about this host, not an answer.
        return (False, None)
    try:
        token = credential_secrets.get_hf_token()
    except Exception:
        return (False, None)
    if isinstance(token, str) and token.strip():
        return (True, token.strip())
    # None has two meanings here and only one of them is an answer. `get_secret` returns it
    # both for an absent row and for a row it could not decrypt -- a lost or rotated
    # encryption key, corrupted ciphertext, a format a newer build wrote -- and reading the
    # second as "this host holds no credential" is a fail-open: the credential-less branch
    # of `_caller_populated_the_cache` would then hand an API-key caller a cache that may
    # hold whatever that unreadable token downloaded.
    try:
        stored = credential_secrets.hf_token_row_exists()
    except Exception:
        return (False, None)
    if stored:
        return (False, None)
    return (True, None)


def _host_hf_credentials() -> "tuple[bool, tuple]":
    """``(known, tokens)``: every HF credential this host holds, from both stores.

    ``known`` is the AND of the two: if either store could not be read, the host's credential
    set is not established and nobody is authorized, on either branch.
    """
    ambient_known, ambient = _ambient_hf_token()
    saved_known, saved = _saved_studio_hf_token()
    if not (ambient_known and saved_known):
        return (False, ())
    return (True, tuple(value for value in (ambient, saved) if value))


# Repos this host fetched with a credential it does not hold. The download route accepts a
# ONE-OFF `X-Unsloth-HF-Token` and hands it to the downloader without saving it anywhere, so
# a private repo can be sitting in the cache of a host whose credential set is empty -- and
# the tokenless branch below reads an empty credential set as "everything here was public
# when it was downloaded". That inference is only sound if no such download ever happened,
# so the ones that did are recorded, per repo, at the moment they are started.
_REQUEST_TOKEN_REPOS_SETTING_KEY = "hub_repos_fetched_with_a_request_token"


def _request_token_repo_key(repo_id: str, repo_type: Optional[str]) -> str:
    return f"{(repo_type or 'model').strip().lower()}:{repo_id.strip()}"


def _as_owner(call, *args, **kwargs):
    """Run a settings read or write as the owner, the way every installation-wide setting is.

    A managed account's own row is not the host's record, and this one is about the host.
    """
    from utils.account_context import OWNER, is_owner_context, run_as

    if is_owner_context():
        return call(*args, **kwargs)
    return run_as(OWNER, call, *args, **kwargs)


def note_repo_fetched_with_a_request_token(
    token: HfTokenArg,
    repo_id: str,
    repo_type: Optional[str] = "model",
) -> None:
    """Record that *repo_id* was fetched under a credential this host does not hold.

    Called at the start of a download. A token that IS one of the host's credentials records
    nothing: the tokenless branch already refuses on a host that holds any credential, and
    the caller presenting that credential is the operator either way. Anonymous downloads
    record nothing either, since a public repo says nothing about anybody.

    Never raises. A record that could not be written is a record that is not there, and the
    read side treats an unreadable store as "cannot say", which refuses.
    """
    if is_anonymous(token) or not isinstance(token, str) or not token or not repo_id:
        return
    try:
        known, host_tokens = _host_hf_credentials()
        if known and any(hmac.compare_digest(token, held) for held in host_tokens):
            return
        from storage.studio_db import upsert_app_setting_map_entry

        _as_owner(
            upsert_app_setting_map_entry,
            _REQUEST_TOKEN_REPOS_SETTING_KEY,
            _request_token_repo_key(repo_id, repo_type),
            {"at": time.time()},
        )
    except Exception:  # noqa: BLE001 -- a download must never fail on its own bookkeeping
        logger.debug("could not record the credential a download used", exc_info = True)


def _repo_was_fetched_with_a_request_token(
    repo_id: Optional[str], repo_type: Optional[str]
) -> Optional[bool]:
    """True / False / ``None`` for "the record could not be read"."""
    if not repo_id:
        return None
    try:
        from storage.studio_db import get_app_setting
        recorded = _as_owner(get_app_setting, _REQUEST_TOKEN_REPOS_SETTING_KEY, None)
    except Exception:  # noqa: BLE001
        return None
    if recorded is None:
        return False  # the store answered, and it holds nothing
    if not isinstance(recorded, dict):
        return None
    return _request_token_repo_key(repo_id, repo_type) in recorded


def _caller_populated_the_cache(
    token: Optional[str],
    *,
    repo_id: Optional[str] = None,
    repo_type: Optional[str] = None,
) -> bool:
    """Whether this caller's credential is the one this host's cache was filled with.

    This is the whole safety of the unaskable fallback. "The repo is on this disk" is a
    fact about the OPERATOR, not about the caller asking, so resolving an unanswerable
    probe against disk presence for any caller at all hands a second principal the
    operator's private downloads: a holder of a valid Studio API key with no permission to
    that HF repo gets it the moment the Hub is offline, times out, 429s, 5xx's or sits
    behind an ``HF_ENDPOINT`` mirror with no /auth-check route. So presence only decides the
    question for a caller that could have produced it.

    A caller with an explicit token qualifies when that token is one the HOST holds: the
    operator's own credential, the one the bytes were fetched with. Nothing is leaked to it
    that it could not fetch from the Hub itself.

    BOTH stores count, and which one matters depends on how Studio was set up. The ambient
    credential (``get_token()``: token file, env aliases, OIDC) is the CLI operator's. The
    Studio UI's is saved in the encrypted credential store and replayed in
    ``X-Unsloth-HF-Token``; it never reaches ``get_token()``. Comparing only against the
    ambient one refuses the ordinary UI case outright, where the operator saved a token in
    Settings, downloaded a private model with it, and configured nothing globally: the
    caller presents the very credential that filled the cache and is told it is not the
    host's. That is the offline flow this gate exists to keep working.

    A caller with NO credential qualifies only when the host holds NO credential in either
    store. Then nothing in the cache can have been fetched under a credential this caller
    lacks, so everything in it was public when it was downloaded. That is the ordinary
    install: most hosts never configure an HF token, which is exactly the offline operator
    this PR exists for. On a host that DOES hold one, a credential-less caller is refused,
    because a private repo could be sitting in that cache.

    Two DIFFERENT held credentials refuse everyone. "One of the host's credentials" is only
    a stand-in for "the credential the bytes were fetched with" while there is one of them;
    with an ambient token and a different saved token, either could have filled the cache,
    and on a managed install the two are different principals. Rather than authorize the
    holder of one for repos the other downloaded, the fallback shuts.

    What this still cannot see is history. The HF cache records no provenance per blob, so
    "the host holds this credential now" is the closest available stand-in for "the bytes
    were fetched under it". An operator who downloads a private repo and then DELETES every
    credential leaves a cache whose contents a credential-less caller is authorized for.
    Closing that needs provenance recorded at download time rather than inferred here, and
    the branch cannot simply be dropped: it is the tokenless offline install, the majority
    case and the reason for the PR.

    Compared with ``compare_digest`` rather than ``==``: the comparison is on a secret, and
    an early-exit compare over a repeated request is a timing oracle for it. Every held
    credential is compared, without short-circuiting on a match, so the time taken does not
    report WHICH store answered.
    """
    known, host_tokens = _host_hf_credentials()
    if not known:
        # Could not be established. Authorize nobody rather than guess, on either branch:
        # guessing "no credential" hands the cache to a caller with none, and guessing
        # "some credential" is not a value a caller's token can be compared against.
        return False
    if token is None:
        if host_tokens:
            return False
        # An empty credential set means nothing in the cache NEEDED one -- unless a download
        # was run with a one-off request token, which the host never keeps. Those repos are
        # recorded, and a record that cannot be read answers nobody.
        return _repo_was_fetched_with_a_request_token(repo_id, repo_type) is False
    if not isinstance(token, str) or not token or not host_tokens:
        return False
    if len({held for held in host_tokens}) > 1:
        # Two DIFFERENT credentials on one host: an ambient one the operator uses from the
        # CLI and a different one saved in Studio's settings by whoever is using the UI.
        # Either could have filled the cache and nothing here can tell which, so matching
        # one of them says nothing about the repo on disk -- and on a managed install the
        # two belong to different principals, which is exactly when saying otherwise hands
        # one of them the other's private downloads. Unknown authorizes nobody. The host
        # that holds a single credential, which is every ordinary install, is unaffected.
        return False
    matched = False
    for held in host_tokens:
        if hmac.compare_digest(token, held):
            matched = True
    return matched


def _resolve_unaskable(repo_id: str, repo_type: str, *, token: Optional[str]) -> bool:
    """What an UNASKABLE Hub means for a cache read. Never called for an answered probe.

    The gate above guards one thing: reading the repo the operator already has on this host.
    So when huggingface.co cannot be reached at all, the question left over is entirely
    local, and the local fact decides it -- BUT only for a caller that fact is about.

    Not this caller's cache -> refused, whatever is on disk. See
    ``_caller_populated_the_cache``: disk presence is a fact about the operator, and
    letting it answer for a second principal is how "could not ask" becomes "here is the
    operator's private repo".

    This caller's cache, on disk -> authorized. The bytes are already here; the operator
    downloaded them with this very credential, and making "can I see my own downloaded
    model" contingent on a round trip to huggingface.co is what broke offline hosts,
    air-gapped installs, Hub outages and ``HF_ENDPOINT`` mirrors that never implemented the
    undocumented /auth-check route.

    Not on disk -> refused, which costs nothing (there is nothing cached to serve) and keeps
    the gate shut for a caller trying to make the backend go and fetch something.

    A Hub that ANSWERS no is untouched by this, which is the boundary the gate was added for:
    online, an API key that cannot reach a private repo is still refused the operator's
    cached copy of it.
    """
    if not _caller_populated_the_cache(token, repo_id = repo_id, repo_type = repo_type):
        return False
    return _repo_present_on_disk(repo_id, repo_type)


def _repo_present_on_disk(repo_id: str, repo_type: str) -> bool:
    """Whether *repo_id* is already materialised in one of this host's HF caches.

    Purely local by construction -- a directory walk, no import of anything that dials out --
    because it is the fallback for "the network is not available" and is reached on every
    call inside the unreachable window, so it must neither block nor be memoized (a download
    completing has to take effect immediately).

    Deliberately ``repo_cache_has_usable_snapshot`` and not "a repo directory exists": an
    interrupted download leaves the directory with no consumable snapshot under it. A dataset
    also counts when only the ``datasets`` PREPARED cache holds it, since that is what backs
    a preview and it is a different tree from the hub snapshot.

    Asked only after a readable listing has actually FOUND a cache dir filed under this repo.
    That ordering is the whole safety of it: ``repo_cache_has_usable_snapshot`` answers True
    when any cache root could not be enumerated, which is right where True means "do not
    delete this" and would here mean every repo is present on a host with one unreadable
    root (a hub cache owned by another user, a stale network mount). Not finding anything is
    not the same as not being able to look, and neither one is presence.
    """
    try:
        from hub.utils.hf_cache_state import (
            iter_repo_cache_dirs,
            repo_cache_has_usable_snapshot,
        )
    except Exception:
        # Cannot establish the local fact -> no local authorization. Deny, do not guess.
        return False
    # The two trees are independent evidence, so the hub answer is COMBINED with the prepared
    # one rather than returned in place of it. A dataset can have a perfectly good prepared
    # cache and a hub repo directory with nothing usable under it -- a pruned snapshot, an
    # interrupted refetch that left only boilerplate -- and returning on the hub answer alone
    # refused a preview the prepared cache can serve in full, which is the offline case this
    # fallback exists for. Not being able to look at the hub tree is not absence either, and
    # it says nothing at all about the prepared one, so that path falls through here too.
    try:
        if any(True for _dir in iter_repo_cache_dirs(repo_type, repo_id)):
            if bool(repo_cache_has_usable_snapshot(repo_type, repo_id)):
                return True
    except Exception:
        import logging
        logging.getLogger(__name__).debug(
            "Could not check the local cache for '%s'", repo_id, exc_info = True
        )
        if repo_type != "dataset":
            return False
    if repo_type != "dataset":
        return False
    try:
        from hub.utils.dataset_cache import latest_processed_dataset_cache_path
        return latest_processed_dataset_cache_path(repo_id) is not None
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


def _cached_repo_access(key: tuple[str, str, str], now: float):
    """The memoized verdict, or ``_CACHE_MISS``.

    Not ``Optional[bool]``: ``None`` is now a verdict of its own ("could not be asked"), so a
    miss needs a value no verdict can take. Returning ``None`` for both re-probed a dead
    endpoint on every call, which is the stall the short TTL exists to prevent.
    """
    cached = _repo_access_cache.get(key)
    if cached is not None and cached[0] > now:
        return cached[1]
    return _CACHE_MISS


def _explicit_token_reaches_repo(
    repo_id: str,
    token: Optional[str],
    repo_type: str,
    offline: bool = False,
) -> Optional[bool]:
    """``True`` reachable, ``False`` the Hub said no, ``None`` the Hub could not be asked."""
    # None asks the public question, under its own key: a public repo answers 200 for every
    # token, so a shared key would let any string claim that verdict.
    key = (
        repo_id.casefold(),
        repo_type,
        hashlib.sha256(token.encode()).hexdigest()[:16] if token else "anonymous",
    )
    cached = _cached_repo_access(key, time.monotonic())
    if cached is not _CACHE_MISS:
        return _with_remembered_denial(key, cached)  # type: ignore[arg-type]
    if offline or _hub_offline():
        # Not a verdict and not memoized: declared offline says nothing about the credential,
        # and a memo would then outlive the moment the network comes back.
        return _with_remembered_denial(key, None)

    with _inflight_lock(key):
        cached = _cached_repo_access(key, time.monotonic())
        if cached is not _CACHE_MISS:
            return _with_remembered_denial(key, cached)  # type: ignore[arg-type]
        try:
            allowed = _probe_repo_access(repo_id, token, repo_type)
        except _ProbeTimedOut:
            allowed = None
        except Exception:
            # The gate's own failure is a failure to ASK, not an answer. Escaping here would
            # 500 the route; returning False would deny a repo that is already on this disk.
            import logging
            logging.getLogger(__name__).debug(
                "Repo access probe for '%s' raised", repo_id, exc_info = True
            )
            allowed = None
        # AFTER the probe: `start + TTL` memoizes an expired entry when the Hub stalls.
        finished = time.monotonic()
        # No elapsed-time rewrite of the verdict. Giving up is recognised by CLASS and by
        # STATUS, not by the clock: every timeout, connect error, proxy failure and transport
        # error already arrives here as None, and the only way `allowed` is False is a
        # classified denial (401, 403, 410, 451, an HF-coded 404, or one of the denial
        # exception classes). Elapsed time cannot tell one of those apart from a slow one --
        # the budget covers the cold `huggingface_hub` import and a distant mirror as well as
        # the request -- and rewriting it to None discarded the Hub's explicit refusal. With
        # no denial remembered, a caller presenting the host's stored but revoked token then
        # passed the caller-populated-the-cache test and was served the private repo.
        expiry = finished + (
            _REPO_ACCESS_UNREACHABLE_TTL_S if allowed is None else _REPO_ACCESS_TTL_S
        )
        with _repo_access_lock:
            if len(_repo_access_cache) >= _REPO_ACCESS_CACHE_MAX:
                _evict_repo_access_locked()
            _repo_access_cache[key] = (expiry, allowed)
        # Only an answer moves the memory. An unaskable Hub leaves whatever the last answer
        # was, which is the whole point of keeping it.
        if allowed is False:
            _remember_denial(key, finished)
        elif allowed is True:
            _forget_denial(key)
    return _with_remembered_denial(key, allowed)


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


def _probe_answer_from_exception(exc: BaseException, response = None) -> Optional[bool]:
    """``False`` when the Hub answered no, ``None`` when it failed to answer at all.

    The old blanket ``except -> False`` is what made an offline host, a Hub outage, a rate
    limit and a mirror without the route indistinguishable from a rejected credential.
    """
    if _is_probe_timeout(exc):
        return None
    for cls in type(exc).__mro__:
        if cls.__name__ in _DENIAL_EXC_NAMES:
            return False
    status = getattr(response, "status_code", None)
    if not isinstance(status, int):
        # hub attaches the response to HfHubHTTPError; a plain httpx error carries it too.
        status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        if status in _DENIAL_STATUSES:
            return False
        if status == 404 and _has_hf_error_code(response):
            # HF's own "no such repo for you". A mirror with no /auth-check route sends a
            # bare 404 instead, which is not an answer about the repo.
            return False
    return None


def _has_hf_error_code(response) -> bool:
    try:
        headers = getattr(response, "headers", None)
        if headers is None:
            return False
        return bool(headers.get("X-Error-Code"))
    except Exception:
        return False


def _probe_repo_access(repo_id: str, token: Optional[str], repo_type: str) -> Optional[bool]:
    """``True`` reachable, ``False`` the Hub answered no, ``None`` it could not be asked."""
    response = None
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
        # hf_raise_for_status passes 3xx: without this a bare 307 reads as authorized. It is
        # not a denial either -- a legacy alias redirects, and so does a captive proxy -- so
        # it is "nobody answered for this repo" and resolves locally.
        if 300 <= (getattr(response, "status_code", 0) or 0) < 400:
            return None
        # And get_session DOES follow them (httpx.Client(follow_redirects=True)), so the
        # check above never fires: a redirect to a login page returns an approving 200 for
        # somewhere else. Only the repo we asked about may answer for it.
        if getattr(response, "history", None):
            return None
        # Belt and braces for a client that does not record history. Compared on parts, not
        # as strings: httpx canonicalises response.url (lower-cases the host, drops an
        # explicit :443) while the configured HF_ENDPOINT keeps its spelling, so a raw
        # comparison denied every mirror written as HF-MIRROR.example or with the port.
        final_url = getattr(response, "url", None)
        if final_url is not None and not _same_probe_target(str(final_url), path):
            return None
        # Last, so the checks above see the response rather than an exception built from it.
        hf_raise_for_status(response)
        return True
    except Exception as exc:
        if _is_probe_timeout(exc):
            raise _ProbeTimedOut from exc
        return _probe_answer_from_exception(exc, response)
