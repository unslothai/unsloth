# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""The offline guard must not outlive the suite that asked for it.

`tests/security/conftest.py` replaces `socket.socket` so a scanner reaching the
internet fails loudly. It was session-scoped: a directory conftest limits WHICH
tests a fixture applies to, but a session-scoped one still tears down when the
session ends, so the patch stayed installed for everything that ran afterwards.

`security` sorts before `version_compat` and `vllm_compat`, whose pinned-symbol
checks fetch upstream sources to detect API drift, so a full run lost about 1300
of them to

    RuntimeError: network access blocked by tests/security/conftest.py

Every one of them passed on its own and failed in the suite, which reads as
upstream drift rather than as a fixture. Pinned here so the scope cannot quietly
go back.
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
