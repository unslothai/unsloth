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
_REPO_ACCESS_CACHE_MAX = 1024
_repo_access_cache: dict[tuple[str, str, str], tuple[float, bool]] = {}
_repo_access_lock = threading.Lock()


def reset_repo_access_cache() -> None:
    """Drop memoized Hub access answers. Tests only."""
    with _repo_access_lock:
        _repo_access_cache.clear()


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
    explicit token is authorized only after it reaches the named repository.
    """
    if is_anonymous(hf_token):
        return False
    if not isinstance(hf_token, str) or not hf_token:
        return True
    repo = (repo_id or "").strip()
    if not repo:
        return False
    return _explicit_token_reaches_repo(repo, hf_token, repo_type)


def _explicit_token_reaches_repo(repo_id: str, token: str, repo_type: str) -> bool:
    key = (
        repo_id.casefold(),
        repo_type,
        hashlib.sha256(token.encode()).hexdigest()[:16],
    )
    now = time.monotonic()
    cached = _repo_access_cache.get(key)
    if cached is not None and cached[0] > now:
        return cached[1]
    allowed = _probe_repo_access(repo_id, token, repo_type)
    with _repo_access_lock:
        if len(_repo_access_cache) >= _REPO_ACCESS_CACHE_MAX:
            _repo_access_cache.clear()
        _repo_access_cache[key] = (now + _REPO_ACCESS_TTL_S, allowed)
    return allowed


def _probe_repo_access(repo_id: str, token: str, repo_type: str) -> bool:
    try:
        from huggingface_hub import HfApi

        HfApi(token = token).repo_info(repo_id, repo_type = repo_type, timeout = 10)
        return True
    except Exception:
        return False
