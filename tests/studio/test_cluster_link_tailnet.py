# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The tailcat tunnel wrapper: against a scripted fake helper, the in-process FakeTailnet, and
(when UNSLOTH_LINK_BIN names a built unsloth-link) two real helpers over a loopback DERP."""

from __future__ import annotations

import json
import os
import socket
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from studio.cluster_link import tailnet

FAKE_HELPER = textwrap.dedent(
    """
    import json, os, sys
    log = open(os.environ["FAKE_LINK_LOG"], "a")
    log.write(json.dumps({"argv": sys.argv[1:]}) + "\\n"); log.flush()
    if sys.argv[1:2] == ["genkey"]:
        args = sys.argv[2:]
        path = args[args.index("--path") + 1]
        with open(path, "w") as f:
            f.write("{}")
        print("nodekey:" + ("c" if "client" in args else "s") * 64)
        sys.exit(0)
    mode = os.environ.get("FAKE_LINK_MODE", "")
    if mode == "crash":
        sys.stderr.write("boom: no DERP\\n"); sys.stderr.flush()
        sys.exit(3)
    print(json.dumps({"event": "ready", "version": 1}), flush = True)
    for line in sys.stdin:
        req = json.loads(line)
        log.write(json.dumps(req) + "\\n"); log.flush()
        op, rid = req["op"], req["id"]
        if op == "shutdown":
            break
        if mode == "silent":
            continue
        if op == "serve":
            out = {"address": "tcSECRET"}
        elif op == "forward":
            out = {"local_port": 40000 + req["remote_port"] % 1000}
        elif op == "probe":
            out = {"direct": req["address"] != "tcRELAY", "rtt_ms": 2.5,
                   "via": "relay sfo" if req["address"] == "tcRELAY" else "direct 1.2.3.4:41641"}
        elif op == "revoke" and req["pubkey"] == "nodekey:bad":
            print(json.dumps({"id": rid, "ok": False, "error": "invalid public key"}), flush = True)
            continue
        else:
            out = {}
        print(json.dumps({"id": rid, "ok": True, **out}), flush = True)
    """
)


@pytest.fixture
def fake_helper(tmp_path, monkeypatch):
    script = tmp_path / "fake_link.py"
    script.write_text(FAKE_HELPER)
    log = tmp_path / "link.log"
    monkeypatch.setenv("FAKE_LINK_LOG", str(log))
    monkeypatch.delenv(tailnet.ENV_DISABLE, raising = False)
    monkeypatch.setattr(tailnet, "_command", lambda: [sys.executable, str(script)])

    def entries():
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines()]

    return entries


def test_helper_path_honours_the_override_and_the_kill_switch(tmp_path, monkeypatch):
    binary = tmp_path / tailnet.BINARY
    monkeypatch.setenv(tailnet.ENV_BIN, str(binary))
    assert tailnet.helper_path() is None and not tailnet.available()
    binary.write_bytes(b"")
    assert tailnet.helper_path() == binary and tailnet.available()
    monkeypatch.setenv(tailnet.ENV_DISABLE, "0")
    assert not tailnet.available()
    with pytest.raises(tailnet.TailnetUnavailable, match = "disabled"):
        tailnet.Tailnet().start()


def test_a_missing_helper_is_unavailable_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.setenv(tailnet.ENV_BIN, str(tmp_path / "nope"))
    monkeypatch.delenv(tailnet.ENV_DISABLE, raising = False)
    with pytest.raises(tailnet.TailnetUnavailable):
        tailnet.Tailnet().start()
    with pytest.raises(tailnet.TailnetUnavailable):
        tailnet.new_client_key(tmp_path / "k")


def test_requests_carry_secrets_on_stdin_only(fake_helper, tmp_path):
    with tailnet.Tailnet(derp_map_url = "https://derp.example/map.json") as net:
        address = net.serve(
            [50070], key_path = tmp_path / "server.key", allow = ["nodekey:" + "a" * 64]
        )
        assert address == "tcSECRET"
        assert net.forward("tcSECRET", 50070, key_path = tmp_path / "client.key") == 40070
        info = net.probe("tcSECRET", timeout_s = 1)
        assert info == tailnet.PathInfo(direct = True, rtt_ms = 2.5, via = "direct 1.2.3.4:41641")
        assert net.probe("tcRELAY", timeout_s = 1).direct is False
        net.allow("nodekey:" + "b" * 64)
        net.revoke("nodekey:" + "b" * 64)
        net.unforward("tcSECRET")
    log = fake_helper()
    assert log[0]["argv"] == ["stdio", "--derpmap-url", "https://derp.example/map.json"]
    assert len([entry for entry in log if "argv" in entry]) == 1
    assert "tcSECRET" not in json.dumps(log[0]) and "client.key" not in json.dumps(log[0])
    ops = [entry["op"] for entry in log if "op" in entry]
    assert ops == ["serve", "forward", "probe", "probe", "allow", "revoke", "unforward"]
    serve = next(entry for entry in log if entry.get("op") == "serve")
    assert serve["allow_all"] is False and serve["key_path"].endswith("server.key")


def test_allow_none_admits_any_address_holder_and_an_empty_list_admits_none(fake_helper):
    with tailnet.Tailnet() as net:
        net.serve([1], key_path = None, allow = None)
        net.serve([2], key_path = None, allow = [])
    serves = [entry for entry in fake_helper() if entry.get("op") == "serve"]
    assert [(s["allow_all"], s["allow"], s["key_path"]) for s in serves] == [
        (True, [], ""),
        (False, [], ""),
    ]


def test_a_refused_request_raises_with_the_helpers_reason(fake_helper):
    with tailnet.Tailnet() as net:
        with pytest.raises(tailnet.TailnetError, match = "invalid public key"):
            net.revoke("nodekey:bad")
        net.allow("nodekey:" + "a" * 64)


def test_a_helper_that_dies_at_start_reports_its_stderr(fake_helper, monkeypatch):
    monkeypatch.setenv("FAKE_LINK_MODE", "crash")
    with pytest.raises(tailnet.TailnetError, match = "no DERP"):
        tailnet.Tailnet().start()


def test_an_unanswered_request_times_out(fake_helper, monkeypatch):
    monkeypatch.setenv("FAKE_LINK_MODE", "silent")
    with tailnet.Tailnet() as net:
        t0 = time.monotonic()
        with pytest.raises(tailnet.TailnetError, match = "did not answer"):
            net._call("allow", timeout = 0.5, pubkey = "nodekey:x")
        assert time.monotonic() - t0 < 5


def test_closing_ends_the_helper(fake_helper):
    net = tailnet.Tailnet()
    net.start()
    proc = net._proc
    net.close()
    assert proc.poll() is not None
    with pytest.raises(tailnet.TailnetError):
        net.allow("nodekey:x")


def test_genkey_writes_through_the_helper(fake_helper, tmp_path):
    assert tailnet.new_server_key(tmp_path / "keys" / "server.key").startswith("nodekey:s")
    assert tailnet.new_client_key(tmp_path / "keys" / "client.key").startswith("nodekey:c")
    argvs = [entry["argv"] for entry in fake_helper() if "argv" in entry]
    assert "--fixed-region" in argvs[0] and "--fixed-region" not in argvs[1]


def _echo_server():
    srv = socket.create_server(("127.0.0.1", 0))

    def run():
        while True:
            try:
                conn, _ = srv.accept()
            except OSError:
                return

            def echo(conn = conn):
                with conn:
                    while True:
                        try:
                            data = conn.recv(65536)
                        except OSError:
                            return
                        if not data:
                            return
                        conn.sendall(data)

            threading.Thread(target = echo, daemon = True).start()

    threading.Thread(target = run, daemon = True).start()
    return srv, srv.getsockname()[1]


def _roundtrip(
    port: int,
    payload: bytes,
    timeout: float = 10.0,
) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout = timeout) as sock:
        sock.sendall(payload)
        got = b""
        while len(got) < len(payload):
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError(f"closed after {len(got)} of {len(payload)} bytes")
            got += chunk
        return got


def test_fake_tailnet_forwards_refuses_strangers_and_revokes():
    srv, port = _echo_server()
    with (
        tailnet.FakeTailnet() as worker,
        tailnet.FakeTailnet() as master,
        tailnet.FakeTailnet() as stranger,
    ):
        address = worker.serve([port], key_path = None, allow = [master.client_pub])
        local = master.forward(address, port)
        assert _roundtrip(local, b"ping") == b"ping"
        with pytest.raises(OSError):
            _roundtrip(stranger.forward(address, port), b"x", timeout = 2)
        live = socket.create_connection(("127.0.0.1", local), timeout = 5)
        live.sendall(b"a")
        assert live.recv(1) == b"a"
        worker.revoke(master.client_pub)
        assert live.recv(1) == b""
        assert master.probe(address).direct
    with pytest.raises(tailnet.TailnetError):
        tailnet.FakeTailnet().probe(address)
    srv.close()


def _real_helper() -> Path:
    path = os.environ.get(tailnet.ENV_BIN) or ""
    if not path or not Path(path).is_file():
        pytest.skip(reason = "set UNSLOTH_LINK_BIN to a built unsloth-link for the real tunnel test")
    return Path(path)


def test_real_helpers_tunnel_refuse_strangers_and_revoke(tmp_path, monkeypatch):
    _real_helper()
    monkeypatch.setenv("TS_DEBUG_TAILCAT_LOCAL_DERP", "1")
    monkeypatch.delenv(tailnet.ENV_DISABLE, raising = False)
    srv, port = _echo_server()
    ours, theirs = tmp_path / "client.key", tmp_path / "stranger.key"
    our_pub = tailnet.new_client_key(ours)
    tailnet.new_client_key(theirs)
    server_key = tmp_path / "server.key"
    server_pub = tailnet.new_server_key(server_key)
    assert server_pub.startswith("nodekey:") and our_pub.startswith("nodekey:")
    if os.name != "nt":
        assert (server_key.stat().st_mode & 0o777) == 0o600
    with tailnet.Tailnet() as worker, tailnet.Tailnet() as master, tailnet.Tailnet() as stranger:
        address = worker.serve([port], key_path = server_key, allow = [our_pub])
        assert address.startswith("tc")
        assert worker.serve([port], key_path = server_key, allow = None) == address
        local = master.forward(address, port, key_path = ours)
        payload = os.urandom(256 * 1024)
        assert _roundtrip(local, payload, timeout = 30) == payload
        info = master.probe(address, timeout_s = 15)
        assert info.direct and info.via.startswith("direct ") and info.rtt_ms is not None

        with pytest.raises(OSError):
            _roundtrip(stranger.forward(address, port, key_path = theirs), b"x", timeout = 5)

        live = socket.create_connection(("127.0.0.1", local), timeout = 10)
        live.sendall(b"a")
        assert live.recv(1) == b"a"
        worker.revoke(our_pub)
        live.settimeout(10)
        try:
            assert live.recv(1) == b""
        except ConnectionResetError:
            pass
        with pytest.raises(OSError):
            _roundtrip(local, b"again", timeout = 5)
    srv.close()


def test_ensure_helper_installs_once_and_falls_back_to_the_lan(monkeypatch):
    from studio.cluster_link import remote

    state = {"installed": False, "calls": 0, "fail": False}

    def install():
        state["calls"] += 1
        if state["fail"]:
            raise tailnet.TailnetError("download refused")
        state["installed"] = True

    notes = []
    monkeypatch.setattr(tailnet, "available", lambda: state["installed"])
    monkeypatch.setattr(tailnet, "install", install, raising = False)
    monkeypatch.setattr(remote, "_HELPER", {})
    assert remote.ensure_helper(notes.append) and remote.ensure_helper(notes.append)
    assert state["calls"] == 1 and notes == []

    monkeypatch.setattr(remote, "_HELPER", {})
    state.update(installed = False, fail = True)
    assert not remote.ensure_helper(notes.append) and not remote.ensure_helper()
    assert state["calls"] == 2 and len(notes) == 1 and "download refused" in notes[0]
    assert "local network only" in notes[0]

    monkeypatch.setattr(remote, "_HELPER", {})
    monkeypatch.setenv("UNSLOTH_CLUSTER_TAILNET", "0")
    assert not remote.ensure_helper(notes.append) and state["calls"] == 2 and len(notes) == 1
