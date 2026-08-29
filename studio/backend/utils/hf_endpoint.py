# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Centralised HuggingFace endpoint configuration.

Backend code that constructs HF URLs directly (i.e. *outside* of
``huggingface_hub`` calls) should use :func:`get_hf_endpoint` instead of
hard-coding ``https://huggingface.co``. The value is read through
:func:`utils.utils.hf_endpoint_url` — the single source of truth for
``HF_ENDPOINT`` parsing — so both entry points stay in lockstep.

The datasets-server base URL is independent from the Hub mirror: most Hub
mirrors do not proxy the datasets-server API, so a mirrored ``HF_ENDPOINT``
never implicitly redirects datasets-server traffic.  Operators who do run a
mirrored datasets-server must set ``HF_DATASETS_SERVER`` explicitly.
"""

from __future__ import annotations

import logging
import os

from utils.utils import hf_endpoint_url

logger = logging.getLogger(__name__)

_DEFAULT_HF_ENDPOINT = "https://huggingface.co"
_DEFAULT_DATASETS_SERVER = "https://datasets-server.huggingface.co"

_ds_mirror_warned = False


def get_hf_endpoint() -> str:
    """Return the configured HuggingFace hub endpoint (no trailing slash).

    Wraps :func:`utils.utils.hf_endpoint_url` so callers get a value that is
    safe for ``f"{endpoint}/path"`` concatenation.
    """
    return hf_endpoint_url().rstrip("/")


def get_hf_datasets_server() -> str:
    """Return the datasets-server base URL (no trailing slash).

    Returns ``HF_DATASETS_SERVER`` when set, otherwise the official
    ``datasets-server.huggingface.co``.  A mirrored ``HF_ENDPOINT`` does
    **not** implicitly apply here — Hub mirrors rarely proxy the
    datasets-server API, so operators must opt in explicitly.
    """
    raw = (os.environ.get("HF_DATASETS_SERVER") or "").strip()
    if raw:
        endpoint = raw if "://" in raw else "https://" + raw
        return endpoint.rstrip("/")
    global _ds_mirror_warned
    if not _ds_mirror_warned and get_hf_endpoint() != _DEFAULT_HF_ENDPOINT:
        _ds_mirror_warned = True
        logger.warning(
            "HF_ENDPOINT is set to %s but HF_DATASETS_SERVER is unset; "
            "datasets-server calls will still go to %s. "
            "Set HF_DATASETS_SERVER to override.",
            get_hf_endpoint(),
            _DEFAULT_DATASETS_SERVER,
        )
    return _DEFAULT_DATASETS_SERVER
