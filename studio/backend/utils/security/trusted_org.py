# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Trusted-organization checks for ``trust_remote_code`` auto-enable paths.

Several load paths auto-enable ``trust_remote_code`` for a narrow set of models
(currently NemotronH/Nano, which need it to work around a transformers
config-parsing bug). The original gate was ``name.startswith("unsloth/")`` /
``startswith("nvidia/")`` on the user-supplied model name. That string check is
spoofable: a *local path* such as ``./unsloth/evil`` or ``/tmp/unsloth/x`` also
starts with ``unsloth/`` and would wrongly be treated as a first-party repo.

``is_trusted_org_repo`` hardens that: it rejects local paths outright, requires
an ``org/repo`` shape whose namespace is a trusted org, and (when online)
confirms via the Hub that the id actually resolves to a repo owned by that org.
It fails CLOSED (returns False) on any uncertainty and never raises - a False
result simply means "do not silently auto-enable remote code; the user can still
opt in via the explicit toggle".
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

from loggers import get_logger
from utils.paths import is_local_path

logger = get_logger(__name__)

# HF organizations whose repos we are willing to auto-enable remote code for.
TRUSTED_ORGS: frozenset[str] = frozenset({"unsloth", "nvidia"})

# Cache keyed on (name, verify_remote, token) so repeated load attempts don't
# re-hit the Hub. The token is part of the key so an unauthenticated failure on a
# private/gated repo can't poison a later authenticated lookup (or vice versa).
_verdict_cache: dict[tuple[str, bool, str], bool] = {}


def _token_key(hf_token: Optional[str]) -> str:
    """Stable, non-reversible discriminator for the cache key. Empty when no
    token; never stores the raw token."""
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
    """Return True only if *name* is a genuine HF repo under a trusted org.

    Fails closed (returns False) for local paths, malformed names, untrusted
    namespaces, and any Hub-verification error. Never raises.

    Offline (``HF_HUB_OFFLINE``/``TRANSFORMERS_OFFLINE``) skips the remote check
    and trusts the namespace shape, since the Hub is unreachable by design.
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
    """Return the lowercased org namespace of an ``org/repo`` id, else None."""
    parts = name.split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0].lower()


def _evaluate(name: str, hf_token: Optional[str], verify_remote: bool) -> bool:
    # 1) Local paths can never be a trusted *remote* repo, even if the path text
    #    happens to start with "unsloth/" (the spoof this check exists to close).
    try:
        if is_local_path(name):
            logger.debug("is_trusted_org_repo(%s): local path -> not trusted", name)
            return False
    except Exception:
        # If we cannot even classify it, be conservative.
        return False

    # 2) Shape + namespace must be a single org/repo under a trusted org.
    ns = _namespace(name)
    if ns is None or ns not in TRUSTED_ORGS:
        return False

    # 3) Offline: trust the shape (Hub is intentionally unreachable).
    if not verify_remote or _env_offline():
        return True

    # 4) Online: confirm the id resolves to a repo owned by a trusted org.
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
    except Exception as exc:  # network error, 404, auth, etc. -> fail closed
        logger.warning(
            "is_trusted_org_repo(%s): Hub verification failed (%s) -> not trusted",
            name,
            exc,
        )
        return False


def clear_cache() -> None:
    """Test helper: drop the memoized verdicts."""
    _verdict_cache.clear()
