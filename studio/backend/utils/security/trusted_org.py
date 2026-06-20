# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Trusted-org checks for the ``trust_remote_code`` auto-enable paths.

A bare ``name.startswith("unsloth/")`` is spoofable by a local path like
``./unsloth/evil``. ``is_trusted_org_repo`` rejects local paths, requires an
``org/repo`` under a trusted org, and (online) confirms via the Hub. Fails CLOSED
on any uncertainty and never raises; a False just means "do not auto-enable".
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

from loggers import get_logger
from utils.paths import is_local_path

logger = get_logger(__name__)

# Orgs we auto-enable remote code for.
TRUSTED_ORGS: frozenset[str] = frozenset({"unsloth", "nvidia"})

# Keyed on (name, verify_remote, token) so an unauthenticated failure can't poison
# a later authenticated lookup; token is hashed, never stored raw.
_verdict_cache: dict[tuple[str, bool, str], bool] = {}


def _token_key(hf_token: Optional[str]) -> str:
    """Non-reversible cache discriminator; empty when no token, never the raw token."""
    if not hf_token:
        return ""
    return hashlib.sha256(hf_token.encode("utf-8")).hexdigest()[:12]


def _env_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes") or os.environ.get(
        "TRANSFORMERS_OFFLINE", ""
    ).lower() in ("1", "true", "yes")


def is_trusted_org_repo(
    name: str,
    hf_token: Optional[str] = None,
    *,
    verify_remote: bool = True,
) -> bool:
    """True only if *name* is a genuine HF repo under a trusted org. Fails closed
    (local paths, malformed names, untrusted namespaces, Hub errors); never raises.
    Offline trusts the namespace shape, since the Hub is unreachable by design.
    """
    if not name or not isinstance(name, str):
        return False

    cache_key = (name, verify_remote, _token_key(hf_token))
    if cache_key in _verdict_cache:
        return _verdict_cache[cache_key]

    verdict = _evaluate(name, hf_token, verify_remote)
    _verdict_cache[cache_key] = verdict
    return verdict


def _namespace(name: str) -> Optional[str]:
    """Lowercased org of an ``org/repo`` id, else None."""
    parts = name.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0].lower()


def _evaluate(name: str, hf_token: Optional[str], verify_remote: bool) -> bool:
    # Local paths are never a trusted remote repo (the spoof this guards against).
    try:
        if is_local_path(name):
            logger.debug("is_trusted_org_repo(%s): local path -> not trusted", name)
            return False
    except Exception:
        return False

    ns = _namespace(name)
    if ns is None or ns not in TRUSTED_ORGS:
        return False

    # Offline: trust the shape (Hub intentionally unreachable).
    if not verify_remote or _env_offline():
        return True

    # Online: confirm the id resolves to a trusted-org repo.
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(name, token = hf_token)
        resolved_id = getattr(info, "id", None) or name
        resolved_ns = _namespace(resolved_id)
        author = getattr(info, "author", None)
        if resolved_ns in TRUSTED_ORGS:
            return True
        if author and str(author).lower() in TRUSTED_ORGS:
            return True
        logger.warning(
            "is_trusted_org_repo(%s): resolved id %r not under a trusted org",
            name,
            resolved_id,
        )
        return False
    except Exception as exc:  # network/404/auth -> fail closed
        logger.warning(
            "is_trusted_org_repo(%s): Hub verification failed (%s) -> not trusted",
            name,
            exc,
        )
        return False


def clear_cache() -> None:
    """Test helper: drop the memoized verdicts."""
    _verdict_cache.clear()
