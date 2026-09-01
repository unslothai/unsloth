# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The offline guard must not outlive the suite that asked for it.

`tests/security/conftest.py` replaces `socket.socket`. It was session-scoped: a
directory conftest limits WHICH tests a fixture applies to, but a session-scoped
one still tears down at session end, so the patch stayed installed for
everything after. `security` sorts before `version_compat` and `vllm_compat`,
whose pinned-symbol checks fetch upstream sources, so a full run lost about 1300
of them to `RuntimeError: network access blocked by tests/security/conftest.py`
-- each passing alone, which reads as upstream drift rather than as a fixture.
"""

from __future__ import annotations

import ast
import socket
from pathlib import Path

import pytest

CONFTEST = Path(__file__).resolve().parent / "conftest.py"


def _network_blocker_decorators():
    tree = ast.parse(CONFTEST.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "network_blocker":
            return node.decorator_list
    pytest.fail("tests/security/conftest.py no longer defines `network_blocker`")


def test_the_blocker_is_not_session_scoped():
    for decorator in _network_blocker_decorators():
        for keyword in getattr(decorator, "keywords", []):
            if keyword.arg != "scope":
                continue
            scope = getattr(keyword.value, "value", None)
            assert scope != "session", (
                "a session-scoped blocker is restored only when the whole run "
                "ends, so every later test that touches the network fails"
            )


def test_the_blocker_is_still_autouse():
    """The other half: it has to apply without each test asking for it."""
    autouse = False
    for decorator in _network_blocker_decorators():
        for keyword in getattr(decorator, "keywords", []):
            if keyword.arg == "autouse":
                autouse = getattr(keyword.value, "value", False) is True
    assert autouse, "the scanner suite must be offline by default"


def test_the_blocker_is_installed_right_now():
    """It is autouse, so this test is already running under it."""
    from tests.security.conftest import _BlockedSocket
    assert socket.socket is _BlockedSocket


def test_an_outbound_connection_is_refused():
    with pytest.raises(RuntimeError, match = "network access blocked"):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("93.184.216.34", 80))


def test_loopback_is_still_allowed():
    """A local server is not the internet, and some scanners use one."""
    from tests.security.conftest import _is_loopback

    assert _is_loopback("127.0.0.1")
    assert _is_loopback("localhost")
    assert not _is_loopback("93.184.216.34")


def test_the_original_socket_is_what_gets_restored():
    """Teardown must hand back the real class, not another blocker: nesting two
    installs and restoring in the wrong order would leave the patch behind."""
    import tests.security.conftest as C
    assert (
        not issubclass(socket.socket, C._BlockedSocket) or socket.socket is C._BlockedSocket
    ), "socket.socket has been wrapped more than once"


def test_the_finalizer_really_restores_the_original():
    """Drive the fixture's own generator, so teardown is observed.

    Every assertion above runs while the fixture is still active, so emptying
    the `finally` leaves them all green while the cross-suite leak returns.
    """
    import tests.security.conftest as C

    # From `_BlockedSocket`'s base, not live: `socket.socket` here is already the blocker, so reading it would compare
    # the patch with itself.
    real = C._BlockedSocket.__bases__[0]
    assert real is not C._BlockedSocket

    # Stand the guard down first:
    outer, socket.socket = socket.socket, real
    try:
        generator = C.network_blocker.__wrapped__()
        next(generator)
        assert socket.socket is C._BlockedSocket, "setup did not install the guard"
        next(generator, None)  # run the finally
        assert socket.socket is real, "teardown did not hand the original back"
    finally:
        socket.socket = outer


def test_a_later_suite_gets_a_working_socket_back(tmp_path):
    """The original bug in miniature, in a nested pytest run: the security
    suite first, an ordinary test after it, and the second must see a real
    socket. A regression fails here rather than 1300 tests away.
    """
    import subprocess
    import sys
    import textwrap

    root = Path(__file__).resolve().parents[2]
    (tmp_path / "security").mkdir()
    (tmp_path / "security" / "conftest.py").write_text(
        textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(root)!r})
        from tests.security.conftest import *          # noqa: F401,F403
        from tests.security.conftest import network_blocker  # noqa: F401
    """),
        encoding = "utf-8",
    )
    (tmp_path / "security" / "test_inside.py").write_text(
        textwrap.dedent("""
        import socket
        def test_the_guard_is_on():
            assert socket.socket.__name__ == "_BlockedSocket"
    """),
        encoding = "utf-8",
    )
    (tmp_path / "test_zafter.py").write_text(
        textwrap.dedent("""
        import socket
        def test_the_guard_is_gone():
            assert socket.socket.__name__ != "_BlockedSocket", (
                "the security suite's socket patch outlived it"
            )
    """),
        encoding = "utf-8",
    )

    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tmp_path),
            "-q",
            "-p",
            "no:randomly",
            "-p",
            "no:cacheprovider",
        ],
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-2000:]
