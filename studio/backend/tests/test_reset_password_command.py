# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The command a locked-out user is told to run has to run.

It is printed exactly when someone cannot get into Unsloth, so every way of
being wrong here is a way of stranding them: naming the executable a policy
denies (issue #8490), or naming an isolated module route that cannot see the
site its own package lives in.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


@pytest.fixture(scope = "module")
def auth():
    """routes/auth.py by path, as test_desktop_auth.py loads it.

    `from routes import auth` runs routes/__init__.py, which imports the whole
    router set and with it structlog, so the module under test would be
    unreachable wherever the full backend stack is not installed.
    """
    route_path = _BACKEND / "routes" / "auth.py"
    spec = importlib.util.spec_from_file_location("_reset_command_auth_route", route_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _spoof_auth_globals(
    monkeypatch,
    auth,
    *,
    os_name = None,
    executable = None,
) -> None:
    """Give routes/auth.py private copies of ``os`` and ``sys``.

    The command builder branches on ``os.name`` and reads ``sys.executable``,
    but the real modules are shared with pytest's terminal reporter and the
    xdist transport, which keep reading them while a test runs. Mutating them
    in-process flipped pathlib to Windows flavour inside pytest's own report
    formatting (``ValueError: ... is not in the subpath of '\\repo\\...'``)
    and killed the worker, taking down or wedging the whole run. Module-private
    copies keep the spoof scoped to the code under test.
    """
    if os_name is not None:
        fake_os = types.ModuleType(os.__name__)
        fake_os.__dict__.update(os.__dict__)
        fake_os.name = os_name
        monkeypatch.setattr(auth, "os", fake_os)
    if executable is not None:
        fake_sys = types.ModuleType(sys.__name__)
        fake_sys.__dict__.update(sys.__dict__)
        fake_sys.executable = executable
        monkeypatch.setattr(auth, "sys", fake_sys)


@pytest.fixture
def windows(monkeypatch, auth):
    _spoof_auth_globals(monkeypatch, auth, os_name = "nt")


def test_a_venv_install_gets_the_isolated_module_route(auth, monkeypatch, windows):
    """-I is right when the package is inside the interpreter's own prefix.

    It drops the working directory from sys.path, so a shell sitting in a
    directory that happens to hold an unsloth_cli folder cannot shadow the
    managed one.
    """
    _spoof_auth_globals(
        monkeypatch,
        auth,
        executable = r"C:\Users\dan\.unsloth\studio\unsloth_studio\Scripts\python.exe",
    )
    monkeypatch.setattr(auth, "_cli_is_inside", lambda _prefix: True)

    command = auth._reset_password_command()

    assert command.endswith("-I -m unsloth_cli studio reset-password")
    assert "unsloth.exe" not in command


def test_a_user_site_install_is_not_told_to_isolate_itself(auth, monkeypatch, windows):
    """-I implies -s, which hides the user site the package is installed in.

    A `pip install --user` install told to run that command gets
    `No module named unsloth_cli`, which is worse than useless to someone who is
    already locked out.
    """
    _spoof_auth_globals(monkeypatch, auth, executable = r"C:\Python313\python.exe")
    monkeypatch.setattr(auth, "_cli_is_inside", lambda _prefix: False)

    command = auth._reset_password_command()

    assert " -I " not in command
    assert "-m unsloth_cli" not in command
    # The bootstrap unsloth_cli/__main__.py documents for exactly this case.
    assert auth._CLI_BOOTSTRAP in command
    assert command.endswith(" studio reset-password")
    # One pair of double quotes wraps it for cmd and PowerShell alike, which
    # only holds while the bootstrap itself carries no double quote.
    assert '"' not in auth._CLI_BOOTSTRAP
    assert command.count('"') == 2
    assert "unsloth.exe" not in command


def test_the_bootstrap_matches_the_one_the_cli_uses(auth):
    """Three copies of this string, and a drift changes argv[0] handling.

    Read from the CLI's own source rather than imported, since the backend may
    be running from a venv that has a different unsloth on it, and via AST
    because the constant there is written as adjacent literals.
    """
    repo_root = _BACKEND.parents[1]
    studio_py = (repo_root / "unsloth_cli" / "commands" / "studio.py").read_text(encoding = "utf-8")
    canonical = None
    for node in ast.walk(ast.parse(studio_py)):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_WINDOWS_CLI_ENTRYPOINT"
            for target in node.targets
        ):
            canonical = ast.literal_eval(node.value)
    assert canonical, "_WINDOWS_CLI_ENTRYPOINT is gone from unsloth_cli/commands/studio.py"
    assert auth._CLI_BOOTSTRAP == canonical


def test_a_spaced_interpreter_path_still_falls_through_to_the_cmd_shim(auth, monkeypatch, windows):
    """Unchanged: a quoted path is written differently in cmd and PowerShell."""
    _spoof_auth_globals(monkeypatch, auth, executable = r"C:\Program Files\Python313\python.exe")
    monkeypatch.setattr(auth, "_cli_is_inside", lambda _prefix: True)

    assert auth._reset_password_command() == "unsloth.cmd studio reset-password"


def test_the_prefix_check_locates_rather_than_imports(auth, tmp_path, monkeypatch):
    """A package outside the prefix must answer False, and a missing one too."""
    inside = tmp_path / "venv" / "Lib" / "site-packages" / "unsloth_cli"
    inside.mkdir(parents = True)
    (inside / "__init__.py").write_text("", encoding = "utf-8")

    class _Spec:
        def __init__(self, origin):
            self.origin = origin

    monkeypatch.setattr(
        auth.importlib.util, "find_spec", lambda _name: _Spec(str(inside / "__init__.py"))
    )
    assert auth._cli_is_inside(str(tmp_path / "venv"))
    assert not auth._cli_is_inside(str(tmp_path / "elsewhere"))

    # A namespace package has no origin, and nothing found returns None. Neither
    # is evidence that isolation would work.
    monkeypatch.setattr(auth.importlib.util, "find_spec", lambda _name: _Spec(None))
    assert not auth._cli_is_inside(str(tmp_path / "venv"))
    monkeypatch.setattr(auth.importlib.util, "find_spec", lambda _name: None)
    assert not auth._cli_is_inside(str(tmp_path / "venv"))


def test_posix_is_untouched(auth, monkeypatch, tmp_path):
    """The console script is what a POSIX user should run; nothing here changes."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "unsloth").write_text("", encoding = "utf-8")
    _spoof_auth_globals(monkeypatch, auth, os_name = "posix", executable = str(bin_dir / "python"))

    assert auth._reset_password_command() == f"{bin_dir / 'unsloth'} studio reset-password"
