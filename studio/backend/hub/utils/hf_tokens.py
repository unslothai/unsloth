# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Request-scoped Hugging Face token helpers."""

from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from contextlib import contextmanager
from typing import Iterable, Literal, MutableMapping, Optional, Union

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
# ``None`` is a THIRD value, not a miss: "the Hub could not be asked", never a denial.
_repo_access_cache: dict[tuple[str, str, str], tuple[float, Optional[bool]]] = {}
_CACHE_MISS = object()
_repo_access_lock = threading.Lock()
# One probe per key: the probe runs outside _repo_access_lock, so a cold key would
# otherwise open a connection per caller.
_repo_access_inflight: dict[tuple[str, str, str], threading.Lock] = {}

# An answered NO never expires and its eviction is remembered, since 429/5xx are "unaskable"
# yet reachable: a refused caller is otherwise one outage from access.
_DENIAL_MEMORY_MAX = 8192
_denied_repo_access: dict[tuple[str, str, str], float] = {}
_denial_memory_is_complete = True


def _remember_denial(key: tuple[str, str, str], now: float) -> None:
    global _denial_memory_is_complete
    with _repo_access_lock:
        _denied_repo_access.pop(key, None)
        while len(_denied_repo_access) >= _DENIAL_MEMORY_MAX:
            _denied_repo_access.pop(next(iter(_denied_repo_access)), None)
            _denial_memory_is_complete = False
        _denied_repo_access[key] = now


def _denial_memory_lost_an_entry() -> bool:
    with _repo_access_lock:
        return not _denial_memory_is_complete


def _forget_denial(key: tuple[str, str, str]) -> None:
    with _repo_access_lock:
        _denied_repo_access.pop(key, None)


def _denial_is_remembered(key: tuple[str, str, str]) -> bool:
    with _repo_access_lock:
        return key in _denied_repo_access


def _with_remembered_denial(key: tuple[str, str, str], verdict: Optional[bool]) -> Optional[bool]:
    """ "Could not ask" reads as the last answer if that was no; once any refusal is evicted, no
    key may claim it was never refused."""
    if verdict is None and (_denial_is_remembered(key) or _denial_memory_lost_an_entry()):
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


# Matched by name, not imported: a version that moves one must not turn a denial into "could not ask".
_DENIAL_EXC_NAMES = frozenset(
    {
        "RepositoryNotFoundError",
        "GatedRepoError",
        "DisabledRepoError",
        "RevisionNotFoundError",
    }
)
# Answers; anything else failed to answer. Not 404: a bare 401 covers private AND missing repos,
# so a 404 means the endpoint has no /auth-check route.
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
    global _denial_memory_is_complete
    with _repo_access_lock:
        _repo_access_cache.clear()
        _repo_access_inflight.clear()
        _denied_repo_access.clear()
        _denial_memory_is_complete = True
    _noted_credential_identities.clear()
    _unrecorded_fetches.clear()


def cache_reads_authorized(
    hf_token: HfTokenArg,
    *,
    repo_id: str,
    repo_type: str = "model",
    offline: bool = False,
) -> bool:
    """Whether this caller may read the host Hub disk cache for *repo_id*.

    Three outcomes, not two: a Hub that ANSWERED no denies, one that could not be ASKED resolves
    against the disk (``_resolve_unaskable``), or a local question depends on reaching
    huggingface.co. ``True`` is not "the token is valid": /auth-check answers 200 for any string
    on a public repo and discriminates on the private and gated ones, which is where cached reads
    are. ``repo_info`` cannot replace it, since gated public metadata returns for an invalid token.
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

    The sentinel never authorizes itself, right for a private repo and wrong for a public one; an
    unauthenticated /auth-check is that difference. Unaskable resolves against the disk.
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

    ``is_cached`` is asked FIRST, so an uncached repo is never probed: refusing one protects
    nothing and costs a legitimate caller its answer whenever the probe is merely unavailable.
    It must fail closed, or the guard's own failure opens the path it guards. "May not read it"
    is not "cannot authorize itself", which is where the sentinel sits.
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


def _env_hf_token() -> "Optional[str]":
    import os
    for key in _HF_TOKEN_ENV_KEYS:
        if key == "HF_OIDC_RESOURCE":
            # Names a token rather than holding one, so it cannot be compared to a caller's.
            continue
        value = os.environ.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _ambient_hf_token() -> "tuple[bool, Optional[str]]":
    """``(False, None)`` is "could not answer"; reading it as "no credential" is a fail-open."""
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
            # It could not answer, but the environment still can, and a credential found
            # there is knowledge rather than a guess.
            env_token = _env_hf_token()
            return (True, env_token) if env_token else (False, None)
        if isinstance(token, str) and token.strip():
            return (True, token.strip())
        # An empty answer is NOT "this host holds nothing": `get_token` reads HF_TOKEN,
        # HUGGING_FACE_HUB_TOKEN, the OIDC exchange and the token file, and nothing else. A
        # credential sitting in an alias this module already honours (HF_HUB_TOKEN,
        # HUGGINGFACE_HUB_TOKEN, HUGGINGFACEHUB_API_TOKEN) read as credentialless, which
        # authorizes a tokenless caller against a cache that credential may have filled.
        return (True, _env_hf_token())
    # No reader at all: the environment is the only thing left to ask, and silence there is
    # "could not answer" rather than "nothing".
    env_token = _env_hf_token()
    return (True, env_token) if env_token else (False, None)


def _saved_studio_hf_token() -> "tuple[bool, Optional[str]]":
    """Never written to the token file, so ``_ambient_hf_token`` cannot see it; unreadable is
    not "none"."""
    try:
        from storage import credential_secrets
    except Exception:
        return (False, None)
    # Value and presence from ONE read: `get_secret` answers None for an absent row AND for an
    # undecryptable one, and conflating them is fail-open, but asking the store twice was two
    # sqlite connections on a path every read of a cached repo goes through.
    try:
        token, stored = credential_secrets.get_hf_token_with_presence()
    except Exception:
        return (False, None)
    if isinstance(token, str) and token.strip():
        return (True, token.strip())
    if stored:
        return (False, None)
    return (True, None)


def _host_hf_credentials() -> "tuple[bool, tuple]":
    """``known`` is the AND of both stores, so an unreadable one authorizes nobody."""
    ambient_known, ambient = _ambient_hf_token()
    saved_known, saved = _saved_studio_hf_token()
    if not (ambient_known and saved_known):
        return (False, ())
    return (True, tuple(value for value in (ambient, saved) if value))


# A one-off `X-Unsloth-HF-Token` is stored nowhere, so an empty credential set would otherwise
# read as "everything here was public". Such fetches are recorded per repo instead.
_REQUEST_TOKEN_REPOS_SETTING_KEY = "hub_repos_fetched_with_a_request_token"


def _request_token_repo_key(repo_id: str, repo_type: Optional[str]) -> str:
    """Case-folded: the authorization cache and `iter_repo_cache_dirs` compare ids that way."""
    return f"{(repo_type or 'model').strip().lower()}:{repo_id.strip().lower()}"


def _as_owner(call, *args, **kwargs):
    from utils.account_context import OWNER, is_owner_context, run_as
    if is_owner_context():
        return call(*args, **kwargs)
    return run_as(OWNER, call, *args, **kwargs)


def note_repo_fetched_with_a_request_token(
    token: HfTokenArg,
    repo_id: str,
    repo_type: Optional[str] = "model",
) -> "Optional[dict]":
    """Record that *repo_id* was fetched under a credential, and WHICH one. Never raises.

    Over-broad on purpose: an extra record withholds bytes the caller can still fetch, a missing
    one leaks a private repo. ``by`` is ``None`` when the fetch cannot be attributed to exactly
    one credential, which authorizes nobody.

    Returns the entry it wrote, for a caller that may have to take it back; None when it wrote
    nothing, or when the entry that stands is an earlier writer's.
    """
    if is_anonymous(token) or not repo_id:
        return None
    if token is not None and (not isinstance(token, str) or not token):
        return None
    try:
        fetched_by: Optional[str] = None
        if token is None:
            known, host_tokens = _host_hf_credentials()
            if known and not host_tokens:
                return None
            if known and len(set(host_tokens)) == 1:
                fetched_by = _credential_identity(host_tokens[0])
        else:
            fetched_by = _credential_identity(token)
        from storage.studio_db import upsert_app_setting_map_entry

        key = _request_token_repo_key(repo_id, repo_type)
        recorded = _recorded_request_token_repos()
        if isinstance(recorded, dict) and key not in recorded:
            # Bounded: the repo id is caller-supplied. A miss in a FULL map means "cannot say".
            # Advisory only, since the map can gain an entry between this read and the write;
            # overshooting the bound by a handful of entries is harmless, refusing to record a
            # fetch is not.
            if len(recorded) >= _REQUEST_TOKEN_REPOS_MAX:
                logger.debug("the request-token provenance map is full; not recording %s", key)
                return None
        # The first record stands, and a second credential claiming what the first filled
        # collapses the attribution. Decided INSIDE the write's transaction: two requests with
        # different credentials for the same uncached repo both read "absent" otherwise, and the
        # last writer then stores its own identity where the truth is "two of them could have".
        entry = {"at": time.time(), "by": fetched_by}
        stored = _as_owner(
            upsert_app_setting_map_entry,
            _REQUEST_TOKEN_REPOS_SETTING_KEY,
            key,
            entry,
            keep_first_writer = True,
            ambiguous_field = "by",
        )
        # Returned so the caller can take it back if the fetch it was written for turns out to
        # have moved nothing; only when the stored entry IS ours, since an earlier writer's
        # record is not this call's to remove.
        if isinstance(stored, dict) and stored.get(key) == entry:
            return entry
    except Exception:  # noqa: BLE001 -- a download must never fail on its own bookkeeping
        logger.debug("could not record the credential a download used", exc_info = True)
        # The read side must not take this missing record for the absence an unfetched repo leaves.
        try:
            _unrecorded_fetches.add(_request_token_repo_key(repo_id, repo_type))
        except Exception:  # noqa: BLE001 -- bookkeeping about bookkeeping, still never raises
            logger.debug("could not note the provenance write that failed", exc_info = True)
    return None


def forget_a_fetch_that_moved_nothing(
    record: "Optional[dict]",
    repo_id: str,
    repo_type: Optional[str] = "model",
) -> None:
    """Take back ``record`` when the call it was written for failed leaving NOTHING on disk.

    Written before the call rather than after it on purpose: a fetch that dies half way has
    still filled the cache, and an unrecorded repo reads later as "nobody needed a credential
    for this". The cost is the other direction -- a 404, a rejected token or an outage leaves a
    record for a fetch that never happened, the first-writer rule keeps it, and a repo a later
    anonymous download fills is then withheld from the tokenless offline caller. So the record
    is taken back only where the disk says the call really did move nothing, and only while it
    is still the record this call wrote.
    """
    if not record or not repo_id:
        return
    try:
        if _repo_present_on_disk(repo_id, repo_type or "model"):
            return
        from storage.studio_db import upsert_app_setting_map_entry

        key = _request_token_repo_key(repo_id, repo_type)
        _as_owner(
            upsert_app_setting_map_entry,
            _REQUEST_TOKEN_REPOS_SETTING_KEY,
            key,
            None,
            delete_if_entry_equals = record,
        )
        _unrecorded_fetches.discard(key)
    except Exception:  # noqa: BLE001 -- bookkeeping, and the over-broad record is the safe side
        logger.debug("could not take back a provenance record", exc_info = True)


@contextmanager
def recording_a_request_token_fetch(
    token: HfTokenArg,
    repo_id: str,
    repo_type: Optional[str] = "model",
):
    """Record the fetch around the call that performs it, and take the record back if that call
    raised without leaving anything on disk. See ``forget_a_fetch_that_moved_nothing``."""
    record = note_repo_fetched_with_a_request_token(token, repo_id, repo_type)
    try:
        yield
    except BaseException:
        forget_a_fetch_that_moved_nothing(record, repo_id, repo_type)
        raise


_REQUEST_TOKEN_REPOS_MAX = 4096

# Provenance writes KNOWN to have failed; a durable marker needs the write that just failed.
# Never evicted, since at the bound the whole set reads as "cannot say".
_UNRECORDED_FETCHES_MAX = 4096
_unrecorded_fetches: "set[str]" = set()


def _provenance_record_is_missing(repo_id: Optional[str], repo_type: Optional[str]) -> bool:
    """A write FAILED, as opposed to nothing having fetched it. Only one of them authorizes."""
    if not repo_id:
        return False
    if len(_unrecorded_fetches) >= _UNRECORDED_FETCHES_MAX:
        return True
    return _request_token_repo_key(repo_id, repo_type) in _unrecorded_fetches


def _recorded_request_token_repos() -> "Optional[dict]":
    try:
        from storage.studio_db import get_app_setting
        recorded = _as_owner(get_app_setting, _REQUEST_TOKEN_REPOS_SETTING_KEY, None)
    except Exception:  # noqa: BLE001
        return None
    if recorded is None:
        return {}
    return recorded if isinstance(recorded, dict) else None


# Credentials this host has EVER held, as one-way digests. Without it, removing one lets a
# tokenless caller inherit its downloads and rotating one lets the successor inherit them.
_HOST_CREDENTIAL_IDENTITIES_SETTING_KEY = "hub_hf_credential_identities_seen"
_HOST_CREDENTIAL_IDENTITIES_MAX = 64
_noted_credential_identities: "set[str]" = set()


def _credential_identity(token: str) -> str:
    """One-way and domain-separated: the ledger must never be readable back into a credential."""
    return hashlib.sha256(
        b"unsloth-hf-credential-identity:" + token.encode("utf-8", "surrogatepass")
    ).hexdigest()


def _host_credential_identities() -> "Optional[dict]":
    """``{}`` when nothing was recorded, ``None`` when unreadable. Unreadable is not "empty"."""
    try:
        from storage.studio_db import get_app_setting
        seen = _as_owner(get_app_setting, _HOST_CREDENTIAL_IDENTITIES_SETTING_KEY, None)
    except Exception:  # noqa: BLE001
        return None
    if seen is None:
        return {}
    return seen if isinstance(seen, dict) else None


def _note_host_credential_identities(tokens: Iterable[str]) -> None:
    """The memo is set only once the identity is KNOWN to be in the ledger: a missing entry
    authorizes MORE, not less."""
    for token in tokens:
        if not isinstance(token, str) or not token:
            continue
        identity = _credential_identity(token)
        if identity in _noted_credential_identities:
            continue
        try:
            from storage.studio_db import upsert_app_setting_map_entry

            seen = _host_credential_identities()
            if seen is None:
                continue
            if identity in seen:
                _noted_credential_identities.add(identity)
                continue
            if len(seen) >= _HOST_CREDENTIAL_IDENTITIES_MAX:
                continue
            _as_owner(
                upsert_app_setting_map_entry,
                _HOST_CREDENTIAL_IDENTITIES_SETTING_KEY,
                identity,
                {"at": time.time()},
            )
            _noted_credential_identities.add(identity)
        except Exception:  # noqa: BLE001
            continue


# A credential this host held but could not read back. Its digest cannot be computed, and
# "could not decrypt it" must not read as "there was never one": the ledger takes this instead,
# which is an identity no caller can ever match, so the fallback refuses.
_UNREADABLE_CREDENTIAL_IDENTITY = "a-credential-this-host-could-not-read"


def note_host_credential_identity(
    token: Optional[str], *, a_credential_was_held: bool = False
) -> None:
    """Enter a credential this host HELD in the ledger, while Studio still has it to hash.

    Called from the save and the delete paths, not only from the authorization gate that reads
    the ledger: on an upgraded install the ledger starts empty, and an operator who replaced or
    cleared the saved token before any unaskable probe ever ran left no trace of the old one at
    all. A later outage then found "this host has never held a credential" and handed a
    tokenless caller everything that credential had downloaded.
    """
    if isinstance(token, str) and token:
        _note_host_credential_identities((token,))
    elif a_credential_was_held:
        _note_host_credential_identities((_UNREADABLE_CREDENTIAL_IDENTITY,))


def _no_other_credential_ever_held(tokens: Iterable[str]) -> Optional[bool]:
    """``None`` when unreadable; a ledger predating a rotation cannot report what it never saw."""
    seen = _host_credential_identities()
    if seen is None:
        return None
    mine = {_credential_identity(token) for token in tokens if isinstance(token, str) and token}
    return not (set(seen) - mine)


def _repo_was_fetched_with_a_request_token(
    repo_id: Optional[str], repo_type: Optional[str]
) -> Optional[bool]:
    if not repo_id:
        return None
    recorded = _recorded_request_token_repos()
    if recorded is None:
        return None
    if _request_token_repo_key(repo_id, repo_type) in recorded:
        return True
    if len(recorded) >= _REQUEST_TOKEN_REPOS_MAX:
        # Past the cap, "not in it" no longer means "not fetched with one". Cannot say.
        return None
    if _provenance_record_is_missing(repo_id, repo_type):
        return None
    return False


def _repo_fetched_by_this_credential(
    token: str, repo_id: Optional[str], repo_type: Optional[str]
) -> bool:
    """``True`` also when there is NO record, the case a cache older than the record exists for."""
    if not repo_id:
        return False
    recorded = _recorded_request_token_repos()
    if recorded is None:
        return False
    entry = recorded.get(_request_token_repo_key(repo_id, repo_type))
    if entry is None:
        if _provenance_record_is_missing(repo_id, repo_type):
            return False
        return len(recorded) < _REQUEST_TOKEN_REPOS_MAX
    if not isinstance(entry, dict):
        return False
    recorded_by = entry.get("by")
    if not isinstance(recorded_by, str) or not recorded_by:
        return False
    return hmac.compare_digest(recorded_by, _credential_identity(token))


def _caller_populated_the_cache(
    token: Optional[str],
    *,
    repo_id: Optional[str] = None,
    repo_type: Optional[str] = None,
) -> bool:
    """The whole safety of the unaskable fallback: disk presence is a fact about the OPERATOR, so
    resolving an unanswerable probe against it for ANY caller hands a second principal the
    operator's private downloads. A cache filled before the record existed reads as "never
    recorded", which authorizes. Compared without short-circuiting, so timing reports nothing."""
    known, host_tokens = _host_hf_credentials()
    if not known:
        return False
    _note_host_credential_identities(host_tokens)
    if token is None:
        if host_tokens:
            return False
        # Held none and never has; a credential since given up must not be inherited tokenless.
        if _no_other_credential_ever_held(()) is not True:
            return False
        return _repo_was_fetched_with_a_request_token(repo_id, repo_type) is False
    if not isinstance(token, str) or not token or not host_tokens:
        return False
    if len({held for held in host_tokens}) > 1:
        # Two DIFFERENT credentials on one host: either could have filled the cache.
        return False
    if _no_other_credential_ever_held(host_tokens) is not True:
        # Holding it now is not having filled the cache with it; otherwise a rotation inherits.
        return False
    matched = False
    for held in host_tokens:
        # Encoded: ``compare_digest`` raises TypeError on a non-ASCII str, and both are user text.
        if hmac.compare_digest(
            token.encode("utf-8", "surrogatepass"), held.encode("utf-8", "surrogatepass")
        ):
            matched = True
    if not matched:
        return False
    # Only the record separates the host credential from a one-off token it does not hold.
    return _repo_fetched_by_this_credential(token, repo_id, repo_type)


def _resolve_unaskable(repo_id: str, repo_type: str, *, token: Optional[str]) -> bool:
    """The local fact decides, but only for a caller it is about; not on disk -> refused."""
    if not _caller_populated_the_cache(token, repo_id = repo_id, repo_type = repo_type):
        return False
    return _repo_present_on_disk(repo_id, repo_type)


def _cache_provenance_is_establishable(repo_id: Optional[str], repo_type: Optional[str]) -> bool:
    """The readers below report not knowing as a plain refusal, misleading a caller weighing one."""
    known, _held = _host_hf_credentials()
    if not known:
        return False
    if _recorded_request_token_repos() is None:
        return False
    return not _provenance_record_is_missing(repo_id, repo_type)


def _denial_can_be_overturned(repo_id: str, repo_type: str, token: Optional[str]) -> bool:
    """Whether remembering this denial protects anything: only where ``_resolve_unaskable`` would
    otherwise say yes. Slots are the point, since the key carries a CALLER-SUPPLIED repo id and
    8192 denials for nonexistent repos force an eviction that refuses every unaskable probe
    process-wide. Unanswerable means remember: not remembering is what loses safety."""
    # BEFORE the rule below, which answers a plain no for a fact it could not establish: a
    # denial dropped during an unreadable store is authorized by the next outage.
    if not _cache_provenance_is_establishable(repo_id, repo_type):
        return True
    try:
        if not _caller_populated_the_cache(token, repo_id = repo_id, repo_type = repo_type):
            return False
    except Exception:  # noqa: BLE001 -- could not establish it; remember, do not discard
        return True
    try:
        return _repo_present_on_disk(repo_id, repo_type)
    except Exception:  # noqa: BLE001 -- same direction
        return True


def _repo_present_on_disk(repo_id: str, repo_type: str) -> bool:
    """Not memoized: a completing download must take effect at once. A dataset also counts when
    only the ``datasets`` PREPARED cache holds it. Asked only after a readable listing FOUND a
    cache dir, since ``repo_cache_has_usable_snapshot`` answers True for an unenumerable root."""
    try:
        from hub.utils.hf_cache_state import (
            iter_repo_cache_dirs,
            repo_cache_has_usable_snapshot,
        )
    except Exception:
        return False
    # Independent evidence: a dataset can have a good prepared cache and an unusable hub dir.
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
    """``None`` is a verdict of its own ("could not be asked"), so a miss needs its own sentinel."""
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
        # Not memoized: a memo would outlive the moment the network comes back.
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
            # The gate's own failure is a failure to ASK, not a denial.
            import logging
            logging.getLogger(__name__).debug(
                "Repo access probe for '%s' raised", repo_id, exc_info = True
            )
            allowed = None
        # AFTER the probe: `start + TTL` memoizes an expired entry when the Hub stalls.
        finished = time.monotonic()
        # No elapsed-time rewrite: the budget covers the cold import, so slow is not denied.
        expiry = finished + (
            _REPO_ACCESS_UNREACHABLE_TTL_S if allowed is None else _REPO_ACCESS_TTL_S
        )
        with _repo_access_lock:
            if len(_repo_access_cache) >= _REPO_ACCESS_CACHE_MAX:
                _evict_repo_access_locked()
            _repo_access_cache[key] = (expiry, allowed)
        if allowed is False:
            if _denial_can_be_overturned(repo_id, repo_type, token):
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
    """A blanket ``except -> False`` cannot tell an outage from a rejected credential."""
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
            # HF's own "no such repo for you"; a mirror with no /auth-check route sends a bare 404.
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
        # hf_raise_for_status passes 3xx, so a bare 307 reads as authorized. Not a denial either.
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
