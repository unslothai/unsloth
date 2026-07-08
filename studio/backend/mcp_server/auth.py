# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Security helpers for the Studio MCP server.

The server runs over **stdio**: a local process spawned by a trusted MCP
client on the same host as Studio. That mirrors Studio's existing loopback
stdio-MCP trust model (see ``utils/host_policy.py``), so no bearer-token auth
is required -- the act of launching ``unsloth studio mcp`` is the opt-in.

Secrets (HuggingFace / Weights & Biases tokens) are resolved from the
environment only, never logged, and may be overridden per tool call by the
caller (an agent) where appropriate.
"""

from __future__ import annotations

import os
from typing import Optional


def resolve_secret(env_var: str, override: Optional[str] = None) -> Optional[str]:
    """Return a secret from an explicit override, else the environment.

    ``override`` (a value passed by the caller/agent) wins, then the named env
    var. Blank strings are treated as absent so a stray ``HF_TOKEN=`` never
    shadows a real configured value.
    """
    if override and override.strip():
        return override.strip()
    env_val = os.environ.get(env_var)
    if env_val and env_val.strip():
        return env_val.strip()
    return None


def resolve_hf_token(override: Optional[str] = None) -> Optional[str]:
    """Resolve a HuggingFace token from ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN``."""
    return resolve_secret("HF_TOKEN", override) or resolve_secret(
        "HUGGING_FACE_HUB_TOKEN", override
    )


# MCP tool hint strings. Surfaced to the client so a human can approve risky
# calls (training, export, uploads) more deliberately than read-only ones.
HINT_READ_ONLY = "Read-only: safe to call freely."
HINT_STATEFUL = "Stateful: starts or mutates a Studio job/resource."
HINT_LONG_RUNNING = (
    "Long-running: may take minutes to hours. The call blocks until the "
    "operation finishes; use the matching *_status tool to poll where available."
)
