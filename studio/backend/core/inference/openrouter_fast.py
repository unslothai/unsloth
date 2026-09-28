# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Discover explicitly advertised fast endpoints; no per-model allowlist or guessed model suffix."""

from typing import Any

from core.inference.provider_model_capabilities import openrouter_pricing


def openrouter_fast_tier(data: Any) -> dict:
    endpoints = data.get("endpoints", []) if isinstance(data, dict) else []
    fast = [
        endpoint
        for endpoint in endpoints
        if isinstance(endpoint, dict)
        and isinstance(endpoint.get("tag"), str)
        and endpoint["tag"].endswith("/fast")
    ]
    active = [endpoint for endpoint in fast if endpoint.get("status") == 0]
    return {
        "supported": bool(fast),
        "available": bool(active),
        "endpoints": [
            {"tag": endpoint["tag"], "pricing": openrouter_pricing(endpoint.get("pricing"))}
            for endpoint in active
        ],
    }
