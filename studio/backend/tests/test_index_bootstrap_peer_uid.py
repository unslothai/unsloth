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


_HDR = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n"


def _row(local, remote, state, uid):
    return f"   0: {local} {remote} {state} 00000000:00000000 00:00000000 00000000 {uid:>5} 0 0\n"


def _tables(
    tmp_path,
    monkeypatch,
    tcp = "",
    tcp6 = "",
):
    import utils.loopback_peer as lp

    paths = []
    for name, body in (("tcp", tcp), ("tcp6", tcp6)):
        p = tmp_path / name
        p.write_text(_HDR + body)
        paths.append(str(p))
    monkeypatch.setattr(lp, "_PROC_TABLES", tuple(paths))
    return lp


def test_a_stale_entry_sharing_only_the_ports_cannot_stand_in_for_the_live_client(
    tmp_path, monkeypatch
):
    # TIME_WAIT from 127.0.0.2 (owned by root) listed before the live 127.0.0.1 client of another user.
    lp = _tables(
        tmp_path,
        monkeypatch,
        tcp = _row("0200007F:C350", "0100007F:22B8", "06", 0)
        + _row("0100007F:C350", "0100007F:22B8", "01", 65534),
    )
    assert lp.loopback_peer_uid(("127.0.0.1", 50000), ("127.0.0.1", 8888)) == 65534
    monkeypatch.setattr(os, "geteuid", lambda: 1000)
    req = _request(("127.0.0.1", 50000), ("127.0.0.1", 8888))
    assert lp.peer_is_another_local_user(req) is True


def test_only_established_entries_count(tmp_path, monkeypatch):
    lp = _tables(tmp_path, monkeypatch, tcp = _row("0100007F:C350", "0100007F:22B8", "06", 0))
    assert lp.loopback_peer_uid(("127.0.0.1", 50000), ("127.0.0.1", 8888)) is None


def test_dual_stack_ipv4_mapped_entry_matches(tmp_path, monkeypatch):
    mapped = "0000000000000000FFFF00000100007F"
    lp = _tables(tmp_path, monkeypatch, tcp6 = _row(f"{mapped}:C350", f"{mapped}:22B8", "01", 4321))
    assert lp.loopback_peer_uid(("127.0.0.1", 50000), ("::ffff:127.0.0.1", 8888)) == 4321
