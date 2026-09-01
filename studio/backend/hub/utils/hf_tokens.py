# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Request-scoped Hugging Face token helpers."""

from __future__ import annotations

from typing import Literal, MutableMapping, Optional, Union

HfTokenArg = Optional[Union[str, Literal[False]]]

# Cache identity for the anonymous sentinel. Every per-token cache key must map ``False``
# to this rather than to the identity it gives ``None``: a slot filled by a UI session
# under the backend's ambient token would otherwise be served straight back to an API key
# that is denied that token, and an API key's anonymous 403 would poison the UI's slot.
# Deliberately not hex, so it can never collide with a sha256 digest of a real token.
ANONYMOUS_CACHE_IDENTITY = "anon"


def hf_token_arg(hf_token: Optional[str], *, allow_ambient_token: bool) -> HfTokenArg:
    """Return the explicit token, or choose ambient versus anonymous access."""
    token = (hf_token or "").strip()
    if token:
        return token
    return None if allow_ambient_token else False


# Every environment variable huggingface_hub will accept a credential from. Mirrors the
# list hub/services/download_lifecycle.py already scrubs for download workers.
_HF_TOKEN_ENV_KEYS = (
    "HF_TOKEN",
    "HF_HUB_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
)


def apply_token_to_child_env(env: MutableMapping[str, str], hf_token: HfTokenArg) -> None:
    """Give a spawned probe exactly the credential its caller is entitled to.

    A child environment is seeded from the parent's, so *not setting* a token is not the
    same as denying one -- the ambient credential is inherited. Only the anonymous
    sentinel scrubs; ``None`` deliberately leaves the inherited environment alone, since
    that caller may use whatever the backend itself has.
    """
    if isinstance(hf_token, str) and hf_token:
        env["HF_TOKEN"] = hf_token
        # An inherited HF_HUB_DISABLE_IMPLICIT_TOKEN=1 would otherwise 401 a gated repo.
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "0"
        return
    if is_anonymous(hf_token):
        for key in _HF_TOKEN_ENV_KEYS:
            env.pop(key, None)
        env["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"


def normalize_token(hf_token: HfTokenArg) -> HfTokenArg:
    """Trim an explicit token without collapsing the anonymous sentinel.

    The idiom ``(hf_token or "").strip() or None`` predates the sentinel and silently
    launders ``False`` into ``None``, which is the value that means "use the backend's
    ambient credential" -- the precise opposite of what the caller asked for.
    """
    if is_anonymous(hf_token):
        return False
    return (hf_token or "").strip() or None


def is_anonymous(hf_token: HfTokenArg) -> bool:
    """True only for the forced-anonymous sentinel, never for a missing token.

    ``hf_token is False`` reads as an identity check on a bool, which invites a
    ``not hf_token`` "simplification" that silently reopens the boundary.
    """
    return hf_token is False
