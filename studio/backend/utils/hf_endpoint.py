# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Centralised HuggingFace endpoint configuration.

Reads ``HF_ENDPOINT`` from the environment (the same variable used by
``huggingface_hub``).  All backend code that constructs HF URLs directly
(i.e. *outside* of ``huggingface_hub`` calls) should use
:func:`get_hf_endpoint` instead of hard-coding ``https://huggingface.co``.
"""

from __future__ import annotations

import os

_DEFAULT_HF_ENDPOINT = "https://huggingface.co"

_raw = os.environ.get("HF_ENDPOINT", "").strip()
_HF_ENDPOINT: str = _raw if _raw else _DEFAULT_HF_ENDPOINT


def get_hf_endpoint() -> str:
    """Return the configured HuggingFace hub endpoint (no trailing slash)."""
    return _HF_ENDPOINT.rstrip("/")
