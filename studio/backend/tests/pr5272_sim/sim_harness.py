"""
Test harness for PR #5272 backend storage.

Mounts studio/backend on sys.path with a stub auth module so we can:
- Call studio_db functions directly
- Spin up a FastAPI app with the chat_history router for HTTP-level tests

Set UNSLOTH_STUDIO_HOME to a tmp dir so the SQLite file is sandboxed.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

# Locate studio/backend relative to this file:
#   .../studio/backend/tests/pr5272_sim/sim_harness.py → .../studio/backend
PR_ROOT = Path(__file__).resolve().parents[2]


def _install_auth_stub() -> None:
    """Inject auth.authentication.get_current_subject without dragging in
    the real auth/storage stack (which needs structlog, etc.)."""
    auth_pkg = types.ModuleType("auth")
    auth_pkg.__path__ = []  # mark as package
    sys.modules.setdefault("auth", auth_pkg)

    authentication = types.ModuleType("auth.authentication")

    # Default subject; tests override per-request via dependency_overrides
    DEFAULT_SUBJECT = "test-subject"

    def get_current_subject() -> str:
        return DEFAULT_SUBJECT

    def get_current_subject_allow_password_change() -> str:
        return DEFAULT_SUBJECT

    authentication.get_current_subject = get_current_subject
    authentication.get_current_subject_allow_password_change = (
        get_current_subject_allow_password_change
    )
    authentication.DEFAULT_SUBJECT = DEFAULT_SUBJECT
    sys.modules["auth.authentication"] = authentication
    auth_pkg.authentication = authentication


def mount() -> tuple[Path, types.ModuleType, types.ModuleType]:
    """Return (tmp_home, studio_db_module, chat_history_router_module)."""
    if str(PR_ROOT) not in sys.path:
        sys.path.insert(0, str(PR_ROOT))

    tmp_home = Path(tempfile.mkdtemp(prefix = "unsloth_studio_test_"))
    os.environ["UNSLOTH_STUDIO_HOME"] = str(tmp_home)

    _install_auth_stub()

    # Drop any cached versions
    for name in list(sys.modules):
        if (
            name.startswith("storage")
            or name.startswith("routes")
            or name.startswith("utils")
        ):
            del sys.modules[name]

    from storage import studio_db  # noqa: E402

    studio_db._schema_ready = False  # force schema re-create in this tmp home

    # Avoid routes/__init__.py importing the whole stack (matplotlib etc.).
    # Use importlib.util to load chat_history directly.
    import importlib.util

    routes_pkg = types.ModuleType("routes")
    routes_pkg.__path__ = [str(PR_ROOT / "routes")]
    sys.modules["routes"] = routes_pkg

    spec = importlib.util.spec_from_file_location(
        "routes.chat_history", str(PR_ROOT / "routes" / "chat_history.py")
    )
    chat_history = importlib.util.module_from_spec(spec)
    sys.modules["routes.chat_history"] = chat_history
    spec.loader.exec_module(chat_history)

    return tmp_home, studio_db, chat_history


def fresh_app():
    """Build a FastAPI app with the chat_history router mounted under /api/chat."""
    from fastapi import FastAPI

    _, studio_db, chat_history = mount()
    app = FastAPI()
    app.include_router(chat_history.router, prefix = "/api/chat")
    return app, studio_db, chat_history


def remove_tmp(home: Path) -> None:
    try:
        shutil.rmtree(home, ignore_errors = True)
    except Exception:
        pass
