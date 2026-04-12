# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Centralised HuggingFace endpoint configuration.

Reads ``HF_ENDPOINT`` from the environment (the same variable used by
``huggingface_hub``).  All backend code that constructs HF URLs directly
(i.e. *outside* of ``huggingface_hub`` calls) should use
:func:`get_hf_endpoint` instead of hard-coding ``https://huggingface.co``.

An optional ``HF_DATASETS_SERVER`` env var overrides the datasets-server
base URL (useful when the mirror only proxies Hub endpoints but not the
datasets-server API).
"""

from __future__ import annotations

import os

_DEFAULT_HF_ENDPOINT = "https://huggingface.co"
_DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co"

_raw = os.environ.get("HF_ENDPOINT", "").strip()
_HF_ENDPOINT: str = (_raw if _raw else _DEFAULT_HF_ENDPOINT).rstrip("/")

_raw_ds = os.environ.get("HF_DATASETS_SERVER", "").strip()
_HF_DATASETS_SERVER: str = (_raw_ds if _raw_ds else "").rstrip("/")


def get_hf_endpoint() -> str:
    """Return the configured HuggingFace hub endpoint (no trailing slash)."""
    return _HF_ENDPOINT


def get_hf_datasets_server() -> str:
    """Return the datasets-server base URL.

    Priority: ``HF_DATASETS_SERVER`` env var > same host as ``HF_ENDPOINT``
    if a mirror is configured > official ``datasets-server.huggingface.co``.
    """
    if _HF_DATASETS_SERVER:
        return _HF_DATASETS_SERVER
    if _HF_ENDPOINT != _DEFAULT_HF_ENDPOINT:
        return _HF_ENDPOINT
    return _DEFAULT_DATASETS_SERVER
