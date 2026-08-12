# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whether the ``datasets`` library is installed in this environment.

Same shape as storage/rag_db.py's RAG_AVAILABLE, and for the same reason: one
install tier deliberately ships without an optional dependency, and every caller
of it has to degrade rather than 500.

The tier is Windows on ARM. pyarrow -- ``datasets``' storage engine -- has never
published a win_arm64 wheel at any version, so on a native ARM64 interpreter the
ARM64 inference-only install drops ``datasets`` entirely (issue #8495). Chat,
model downloads and the Images/Video pages do not touch it: the import graph
reachable from main.py contains neither ``datasets`` nor ``pyarrow``. Training,
Hub dataset previews and Data Recipes do, and answer 503 here.

This module must stay importable with ``datasets`` absent, so it never imports
it -- ``find_spec`` only. Importing ``datasets`` costs seconds and pulls pyarrow,
and this is read on request paths.

Not a substitute for the ordinary lazy imports elsewhere: those still raise
ModuleNotFoundError if reached. This is the gate that keeps them unreached.
"""

from __future__ import annotations

import importlib.util
import logging
import os

logger = logging.getLogger(__name__)

_UNAVAILABLE_MSG = (
    "Datasets are unavailable in this installation: the datasets library (and pyarrow, "
    "which it is built on) publishes no wheels for native ARM64 Windows. Training, "
    "dataset previews and Data Recipes need it; chat and model downloads do not. "
    "Install x64 Python from https://www.python.org/downloads/windows/ (it runs "
    "emulated) and re-run the Unsloth installer to get the full product."
)


def _probe() -> bool:
    # find_spec, not import: an absent datasets is the expected state here, and
    # importing it to find out would cost a multi-second pyarrow load on every
    # environment that does have it.
    try:
        return importlib.util.find_spec("datasets") is not None
    except (ImportError, ValueError):
        # ValueError: a namespace-package shadow leaves __spec__ None. Either way
        # the library cannot be used, which is what this answers.
        return False


DATASETS_AVAILABLE: bool = _probe()

if not DATASETS_AVAILABLE:
    logger.warning(
        "datasets is not installed: training, dataset previews and Data Recipes are "
        "unavailable in this installation (inference is not affected)",
    )


class DatasetsUnavailable(RuntimeError):
    """``datasets`` is not installed in this environment.

    Subclasses RuntimeError so existing ``except RuntimeError`` callers keep
    working; it exists so a caller can tell "this install has no datasets" from a
    real dataset error and degrade instead of returning 500 on every poll.
    """

    def __init__(self, message: str = _UNAVAILABLE_MSG) -> None:
        super().__init__(message)


def datasets_available() -> bool:
    """Whether datasets can be imported here.

    A function as well as the constant, mirroring rag_db.rag_available(): tests and
    the tier's own installer flip the environment underneath a long-lived process,
    and UNSLOTH_FORCE_NO_DATASETS makes the unavailable path reachable on a machine
    that does have the library.
    """
    if os.environ.get("UNSLOTH_FORCE_NO_DATASETS") == "1":
        return False
    return DATASETS_AVAILABLE


def require_datasets() -> None:
    """Raise DatasetsUnavailable unless datasets is installed."""
    if not datasets_available():
        raise DatasetsUnavailable()


def unavailable_detail() -> str:
    """The user-facing reason, for a 503 body or the health payload."""
    return _UNAVAILABLE_MSG


def require_datasets_http() -> None:
    """FastAPI dependency: 503 with a stated reason when datasets is absent.

    Mirrors routes/rag.py's _require_rag -- a 503 the UI can read, instead of the
    500 plus traceback a bare ModuleNotFoundError from a lazy ``from datasets
    import ...`` deeper in the call produces on every poll.
    """
    if not datasets_available():
        # Imported here rather than at module scope: this module is also read by
        # the installer-side tests, which have no FastAPI.
        from fastapi import HTTPException

        raise HTTPException(status_code = 503, detail = _UNAVAILABLE_MSG)
