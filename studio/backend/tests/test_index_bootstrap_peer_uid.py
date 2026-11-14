# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The first-run password is not served to a loopback connection owned by another OS user."""

import os
import socket
import sys
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or not os.path.exists("/proc/net/tcp"),
    reason = "peer UID lookup is Linux-only",
)


@pytest.fixture
def loopback_pair():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    cli = socket.create_connection(srv.getsockname())
    conn, peer = srv.accept()
    try:
        yield peer, srv.getsockname()
    finally:
        for s in (cli, conn, srv):
            s.close()


def _request(peer, server):
    host = f"{server[0]}:{server[1]}"
    return SimpleNamespace(
        client = SimpleNamespace(host = peer[0], port = peer[1]),
        headers = {"host": host},
        url = SimpleNamespace(scheme = "http", netloc = host, hostname = server[0]),
        scope = {"server": server},
    )


def test_same_user_loopback_browser_still_gets_the_autofill(loopback_pair):
    from main import _should_inject_bootstrap
    assert _should_inject_bootstrap(_request(*loopback_pair)) is True


def test_other_os_user_on_loopback_does_not_get_the_password(loopback_pair, monkeypatch):
    from main import _should_inject_bootstrap

    # The real peer socket is owned by our UID; pretend the server runs as someone else.
    real = os.geteuid()
    if real == 0:
        pytest.skip("root peers are allowed by design")
    monkeypatch.setattr(os, "geteuid", lambda: real + 4242)
    assert _should_inject_bootstrap(_request(*loopback_pair)) is False


def test_unknown_peer_keeps_todays_behaviour():
    from main import _should_inject_bootstrap
    req = _request(("127.0.0.1", 1), ("127.0.0.1", 2))
    assert _should_inject_bootstrap(req) is True
