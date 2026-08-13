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

import importlib
import importlib.util
import logging
import os
import sys
import sysconfig
from pathlib import Path

logger = logging.getLogger(__name__)

_WHAT_IS_LOST = (
    "Training, dataset previews and Data Recipes need it; chat and model downloads do not."
)

# The ARM64-Windows story is the REASON this gate exists, but it is not the only
# way to arrive here: any environment missing `datasets` lands in the same branch,
# including an ordinary Linux install with a half-finished venv. Telling that user
# to "install x64 Python from python.org (it runs emulated)" is advice for a
# machine they are not sitting at, and it hides the actual remedy. So the tier
# gets its own text and everyone else gets the plain one.
_ARM64_WINDOWS_MSG = (
    "Datasets are unavailable in this installation: the datasets library (and pyarrow, "
    "which it is built on) publishes no wheels for native ARM64 Windows. "
    + _WHAT_IS_LOST
    + " Install x64 Python from https://www.python.org/downloads/windows/ (it runs "
    "emulated) and re-run the Unsloth installer to get the full product."
)

_GENERIC_MSG = (
    "Datasets are unavailable in this installation: the datasets library is not "
    "installed. " + _WHAT_IS_LOST + " Re-run the Unsloth installer, or install it "
    "into this environment with `pip install datasets`."
)


def _is_arm64_windows() -> bool:
    """The tier this gate was built for: a NATIVE ARM64 Python on Windows.

    `sysconfig.get_platform()` rather than `platform.machine()`, because the
    question is which wheels this interpreter can install, and an x64 interpreter
    emulated on ARM hardware answers "win-amd64" here -- correctly, since it can
    install pyarrow. Same predicate the installer keys the tier off.
    """
    if sys.platform != "win32":
        return False
    try:
        return sysconfig.get_platform() == "win-arm64"
    except Exception:
        return False


_UNAVAILABLE_MSG = _ARM64_WINDOWS_MSG if _is_arm64_windows() else _GENERIC_MSG


def _spec_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        # ValueError: a namespace-package shadow leaves __spec__ None. Either way
        # the library cannot be used, which is what this answers.
        return False


def _probe() -> bool:
    # find_spec, not import: an absent datasets is the expected state here, and
    # importing it to find out would cost a multi-second pyarrow load on every
    # environment that does have it.
    #
    # pyarrow as well as datasets, because the two go missing independently. This
    # tier exists precisely because pyarrow has no win_arm64 wheel, and a pass that
    # installs datasets and then dies building pyarrow (or an environment migrated
    # out of the tier halfway) leaves the distribution on disk with its storage
    # engine gone. `import datasets` pulls pyarrow eagerly -- datasets/__init__.py
    # reaches arrow_dataset -- so find_spec("datasets") alone answers True for an
    # environment where the very first `from datasets import ...` still raises
    # ModuleNotFoundError, which is the 500 this gate is here to replace.
    return _spec_present("datasets") and _spec_present("pyarrow")


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
    if not DATASETS_AVAILABLE:
        # Re-probed, because the 503 tells the user to install it: without this the
        # gate stays shut for the life of the process and the advice does nothing
        # until Studio is restarted. Two find_spec calls, on the failing path only.
        #
        # The answer is deliberately NOT written back: a probe taken while `pip
        # install datasets` is still unpacking can see the spec before the package
        # works, and latching that would swap this gate's 503 for a permanent 500.
        # Unlatched, the next request asks again and a half-finished install closes
        # the gate again by itself.
        importlib.invalidate_caches()
        return _probe()
    return DATASETS_AVAILABLE


def require_datasets() -> None:
    """Raise DatasetsUnavailable unless datasets is installed."""
    if not datasets_available():
        raise DatasetsUnavailable()


def unavailable_detail() -> str:
    """The user-facing reason, for a 503 body or the health payload."""
    return _UNAVAILABLE_MSG


def is_inference_only_tier() -> bool:
    """Is this the ARM64 inference-only INSTALL, rather than merely a venv that
    happens to be missing ``datasets``?

    The distinction decides one thing only: whether /api/health reports the whole
    host as chat-only. `chat_only` means "this device can serve GGUF chat and
    nothing else" -- the frontend hides safetensors models, Train, Video and the
    Hub's Run button on it. That is the truth on the tier, which is a torch-less
    ARM64 install. It is NOT the truth on an ordinary Linux or x64 Windows box
    whose venv lost `datasets` to a half-finished update: that host still has its
    GPU, still runs safetensors, and answering chat_only there would take features
    away from a machine this change is not supposed to touch at all.

    So the health verdict keys off the tier, while the 503 route gates key off the
    library, which is the right question for each.

    The tier is recognised the same two ways the installer records it: a native
    ARM64 interpreter (which cannot install pyarrow at any version), or the marker
    install_manifest writes, which also survives an interrupted pass.
    """
    if datasets_available():
        return False
    if os.environ.get("UNSLOTH_FORCE_NO_DATASETS") == "1":
        # The test and reproduction hook: forcing the library absent forces the tier
        # with it, so the degraded UI can be exercised off ARM64 Windows.
        return True
    if _is_arm64_windows():
        return True
    try:
        return (Path(sys.prefix) / ".unsloth-no-datasets").exists()
    except OSError:
        return False


async def require_datasets_http() -> None:
    """FastAPI dependency: 503 with a stated reason when datasets is absent.

    Mirrors routes/rag.py's _require_rag -- a 503 the UI can read, instead of the
    500 plus traceback a bare ModuleNotFoundError from a lazy ``from datasets
    import ...`` deeper in the call produces on every poll.

    ``async`` although it does no I/O: FastAPI runs a *synchronous* dependency in
    the AnyIO worker pool, so on a healthy install where this only reads a bool,
    every gated request would still have to win a worker token before it could
    even be rejected as unauthenticated. Under load from the sync routes elsewhere
    in this app that is a queue in front of a dict lookup. Async runs it on the
    event loop, which is what the work actually costs.
    """
    if not datasets_available():
        # Imported here rather than at module scope: this module is also read by
        # the installer-side tests, which have no FastAPI.
        from fastapi import HTTPException
        raise HTTPException(status_code = 503, detail = _UNAVAILABLE_MSG)
