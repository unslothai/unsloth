# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""On-demand install of the optional RAG Python dependencies.

The no-torch Studio install omits the RAG group, so RAG starts disabled. On
first use while unavailable, ``ensure_async()`` pip-installs the pinned group
(matching ``studio.txt``) in the background and re-enables RAG via
``rag_db.refresh_rag_available()`` without a restart.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import shutil
import subprocess
import sys
import threading

logger = logging.getLogger(__name__)

# (pip requirement spec, import name)
_RAG_PACKAGES = (
    ("sqlite-vec==0.1.9", "sqlite_vec"),
    ("pymupdf==1.27.2.3", "pymupdf"),
    ("python-docx==1.2.0", "docx"),
)

_INSTALL_TIMEOUT_S = 600

_lock = threading.Lock()
_installing = False
_error: str | None = None


def missing_packages() -> list[str]:
    """Requirement specs whose import is not resolvable in this interpreter."""
    return [req for req, mod in _RAG_PACKAGES if importlib.util.find_spec(mod) is None]


def status() -> dict:
    from storage import rag_db
    return {
        "available": rag_db.RAG_AVAILABLE,
        "installing": _installing,
        "missing": missing_packages(),
        "error": _error,
    }


def _pip_install_cmd(*args: str) -> list[str]:
    """`uv pip install` if uv is on PATH, else `python -m pip install`."""
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable, *args]
    return [sys.executable, "-m", "pip", "install", *args]


def _install(reqs: list[str]) -> None:
    global _installing, _error
    try:
        logger.info("Installing RAG dependencies: %s", " ".join(reqs))
        result = subprocess.run(
            _pip_install_cmd(*reqs),
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            timeout = _INSTALL_TIMEOUT_S,
        )
        if result.returncode != 0:
            _error = f"pip install failed: {(result.stdout or '').strip()[-500:]}"
            logger.warning("RAG dependency install failed:\n%s", result.stdout)
            return
        importlib.invalidate_caches()
        from storage import rag_db

        rag_db.refresh_rag_available()
        _error = None
        logger.info("RAG dependencies installed; available=%s", rag_db.RAG_AVAILABLE)
    except subprocess.TimeoutExpired:
        _error = "RAG dependency install timed out."
        logger.warning(_error)
    except Exception as exc:  # noqa: BLE001 - report any install failure, never crash
        _error = f"{type(exc).__name__}: {exc}"
        logger.warning("RAG dependency install error: %s", exc)
    finally:
        _installing = False


def ensure_async(*, force: bool = False) -> None:
    """Start a background install of any missing RAG deps. Idempotent, non-blocking.

    No-op if RAG is available or an install is running. A prior failure is
    sticky so the status poller cannot respin pip; ``force`` (Retry) clears it
    and re-attempts. If the deps are already present, just re-checks the import.
    """
    from storage import rag_db

    if rag_db.RAG_AVAILABLE:
        return
    global _installing, _error
    with _lock:
        if _installing or (_error is not None and not force):
            return
        missing = missing_packages()
        if not missing:
            rag_db.refresh_rag_available()
            return
        _installing, _error = True, None
    threading.Thread(target = _install, args = (missing,), name = "rag-deps-install", daemon = True).start()
