# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's Landlock ruleset scopes abstract AF_UNIX sockets as well as signals."""

import os
import socket
import subprocess
import sys

import pytest

from core.inference import tool_confinement


def test_the_managed_ruleset_scopes_abstract_unix_sockets(monkeypatch):
    monkeypatch.setattr(tool_confinement, "landlock_abi", lambda: 6)
    monkeypatch.setattr(tool_confinement, "_landlock_rules", lambda abi, site: [])
    confinement = tool_confinement._linux_confinement("/tmp/site")
    assert confinement is not None
    scope = confinement.preexec.args[2]
    abstract_unix = 1 << 0  # LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET
    signal = 1 << 1  # LANDLOCK_SCOPE_SIGNAL
    assert scope & abstract_unix
    assert scope & signal


_live = pytest.mark.skipif(
    sys.platform != "linux" or tool_confinement.landlock_abi() < 6,
    reason = "needs a kernel able to apply the ABI 6 scopes",
)

_CONNECT = (
    "import socket, sys\n"
    "s = socket.socket(socket.AF_UNIX)\n"
    "try:\n"
    "    s.connect(sys.argv[1].replace('@', chr(0), 1)); print('CONNECTED')\n"
    "except OSError as e:\n"
    "    print('refused', e.errno)\n"
)


def _confined_connect(monkeypatch, address):
    # Read-only `/` stands in for the account's rules, which need a configured Unsloth Studio home.
    tc = tool_confinement
    read_only = tc._FS_EXECUTE | tc._FS_READ_FILE | tc._FS_READ_DIR
    monkeypatch.setattr(tc, "_landlock_rules", lambda abi, site: [("/", read_only)])
    confinement = tool_confinement._linux_confinement("/tmp/site")
    assert confinement is not None
    return subprocess.run(
        [sys.executable, "-c", _CONNECT, address],
        preexec_fn = confinement.preexec,
        capture_output = True,
        text = True,
        timeout = 30,
    ).stdout.strip()


@_live
def test_a_confined_child_cannot_reach_an_outside_abstract_socket(monkeypatch):
    name = f"unsloth-abstract-probe-{os.getpid()}"
    server = socket.socket(socket.AF_UNIX)
    server.bind("\0" + name)
    server.listen(1)
    try:
        assert _confined_connect(monkeypatch, "@" + name).startswith("refused")
    finally:
        server.close()
