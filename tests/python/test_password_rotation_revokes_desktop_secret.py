"""update_password must revoke the local desktop secret so rotation is effective."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
STORAGE_PY = REPO_ROOT / "studio" / "backend" / "auth" / "storage.py"
HASHING_PY = REPO_ROOT / "studio" / "backend" / "auth" / "hashing.py"


def _stub_sys_modules(tmp_path):
    class _Logger:
        def __getattr__(self, name):
            return self
        def __call__(self, *a, **k):
            return self
        def info(self, *a, **k): pass
        def debug(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
    sys.modules.setdefault(
        "structlog",
        types.SimpleNamespace(get_logger=lambda *a, **k: _Logger()),
    )

    def _auth_db_path():
        return tmp_path / "auth.db"

    def _ensure_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)

    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    paths_mod = types.ModuleType("utils.paths")
    paths_mod.auth_db_path = _auth_db_path
    paths_mod.ensure_dir = _ensure_dir
    sys.modules["utils"] = utils_pkg
    sys.modules["utils.paths"] = paths_mod

    class _Dice:
        @staticmethod
        def handle_options(args):
            return object()
        @staticmethod
        def get_passphrase(options):
            return "stubpassphrase"
    sys.modules.setdefault("diceware", _Dice)


def _fresh_storage(tmp_path):
    """Load storage.py in isolation pointing at tmp_path/auth.db."""
    _stub_sys_modules(tmp_path)
    sys.modules.pop("auth", None)
    sys.modules.pop("auth.storage", None)
    sys.modules.pop("auth.hashing", None)

    auth_pkg = types.ModuleType("auth")
    auth_pkg.__path__ = [str(STORAGE_PY.parent)]
    auth_pkg.__package__ = "auth"
    sys.modules["auth"] = auth_pkg

    spec_h = importlib.util.spec_from_file_location("auth.hashing", HASHING_PY)
    hashing = importlib.util.module_from_spec(spec_h)
    sys.modules["auth.hashing"] = hashing
    spec_h.loader.exec_module(hashing)

    spec_s = importlib.util.spec_from_file_location("auth.storage", STORAGE_PY)
    storage = importlib.util.module_from_spec(spec_s)
    sys.modules["auth.storage"] = storage
    spec_s.loader.exec_module(storage)
    storage.DB_PATH = tmp_path / "auth.db"
    storage._BOOTSTRAP_PW_PATH = tmp_path / ".bootstrap_password"
    return storage


def test_update_password_invokes_clear_desktop_secret(tmp_path, monkeypatch):
    storage = _fresh_storage(tmp_path)
    storage.ensure_default_admin()

    secret = storage.create_desktop_secret()
    assert storage.validate_desktop_secret(secret) == storage.DEFAULT_ADMIN_USERNAME

    ok = storage.update_password(storage.DEFAULT_ADMIN_USERNAME, "new-strong-passphrase")
    assert ok is True

    # Old desktop secret must no longer grant admin tokens.
    assert storage.validate_desktop_secret(secret) is None


def test_update_password_does_not_clear_desktop_secret_when_user_missing(tmp_path):
    storage = _fresh_storage(tmp_path)
    storage.ensure_default_admin()

    secret = storage.create_desktop_secret()
    assert storage.validate_desktop_secret(secret) == storage.DEFAULT_ADMIN_USERNAME

    # No-op update (unknown user) must NOT wipe the legitimate secret.
    ok = storage.update_password("ghost-user", "irrelevant")
    assert ok is False
    assert storage.validate_desktop_secret(secret) == storage.DEFAULT_ADMIN_USERNAME


def test_source_calls_clear_desktop_secret_on_rowcount_branch():
    src = STORAGE_PY.read_text()
    func_start = src.index("def update_password(")
    func_end = src.index("\n\n\n", func_start)
    body = src[func_start:func_end]
    assert "clear_bootstrap_password()" in body
    assert "clear_desktop_secret()" in body
    # The desktop-secret clear must be inside the rowcount>0 branch, not unconditional.
    bootstrap_line = body.index("clear_bootstrap_password()")
    desktop_line = body.index("clear_desktop_secret()")
    rowcount_line = body.index("cursor.rowcount > 0")
    assert rowcount_line < bootstrap_line
    assert rowcount_line < desktop_line
