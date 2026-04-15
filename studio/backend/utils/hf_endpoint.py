# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Centralised HuggingFace endpoint configuration.

Reads ``HF_ENDPOINT`` from the environment (the same variable used by
``huggingface_hub``).  All backend code that constructs HF URLs directly
(i.e. *outside* of ``huggingface_hub`` calls) should use
:func:`get_hf_endpoint` instead of hard-coding ``https://huggingface.co``.

The datasets-server base URL is independent from the Hub mirror: most Hub
mirrors do not proxy the datasets-server API, so a mirrored ``HF_ENDPOINT``
never implicitly redirects datasets-server traffic.  Operators who do run a
mirrored datasets-server must set ``HF_DATASETS_SERVER`` explicitly.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DEFAULT_HF_ENDPOINT = "https://huggingface.co"
_DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co"

_raw = os.environ.get("HF_ENDPOINT", "").strip()
_HF_ENDPOINT: str = (_raw if _raw else _DEFAULT_HF_ENDPOINT).rstrip("/")

_raw_ds = os.environ.get("HF_DATASETS_SERVER", "").strip()
_HF_DATASETS_SERVER: str = (_raw_ds if _raw_ds else "").rstrip("/")

if _HF_ENDPOINT != _DEFAULT_HF_ENDPOINT and not _HF_DATASETS_SERVER:
    logger.warning(
        "HF_ENDPOINT is set to %s but HF_DATASETS_SERVER is unset; "
        "datasets-server calls will still go to %s. "
        "Set HF_DATASETS_SERVER to override.",
        _HF_ENDPOINT,
        _DEFAULT_DATASETS_SERVER,
    )


def get_hf_endpoint() -> str:
    """Return the configured HuggingFace hub endpoint (no trailing slash)."""
    return _HF_ENDPOINT


def get_hf_datasets_server() -> str:
    """Return the datasets-server base URL (no trailing slash).

    Returns ``HF_DATASETS_SERVER`` when set, otherwise the official
    ``datasets-server.huggingface.co``.  A mirrored ``HF_ENDPOINT`` does
    **not** implicitly apply here — Hub mirrors rarely proxy the
    datasets-server API, so operators must opt in explicitly.
    """
    if _HF_DATASETS_SERVER:
        return _HF_DATASETS_SERVER
    return _DEFAULT_DATASETS_SERVER
