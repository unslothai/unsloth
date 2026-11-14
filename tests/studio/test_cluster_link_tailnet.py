# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The tailcat tunnel wrapper: against a scripted fake helper, the in-process FakeTailnet, and
(when UNSLOTH_LINK_BIN names a built unsloth-link) two real helpers over a loopback DERP."""

from __future__ import annotations

import json
import os
import socket
import subprocess
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
    if mode == "exit":
        sys.exit(0)
    if mode == "stuck":
        import time
        time.sleep(120)
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


def test_a_relative_override_names_the_file_it_validated(tmp_path, monkeypatch):
    (tmp_path / tailnet.BINARY).write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(tailnet.ENV_DISABLE, raising = False)
    monkeypatch.setenv(tailnet.ENV_BIN, f"./{tailnet.BINARY}")
    assert tailnet.helper_path() == tmp_path / tailnet.BINARY
    assert tailnet._command() == [str(tmp_path / tailnet.BINARY)]


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
    argv = ["stdio", "--derpmap-url", "https://derp.example/map.json"]
    if os.name != "nt":
        argv += ["--parent-pid", str(os.getpid())]
    assert log[0]["argv"] == argv
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


def test_a_call_racing_close_never_starts_another_helper(fake_helper):
    net = tailnet.Tailnet()
    net.start()
    proc = net._proc
    net.close()
    with pytest.raises(tailnet.TailnetError):
        net.allow("nodekey:x")
    assert proc.poll() is not None and net._proc is None
    assert len([entry for entry in fake_helper() if "argv" in entry]) == 1


def test_a_helper_that_exits_is_reaped_and_reported_dead(fake_helper, monkeypatch):
    monkeypatch.setenv("FAKE_LINK_MODE", "exit")
    net = tailnet.Tailnet()
    net.start()
    proc = net._proc
    deadline = time.monotonic() + 10
    while proc.returncode is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert proc.returncode == 0 and proc.stdout.closed
    assert net.dead
    with pytest.raises(tailnet.TailnetError, match = "exited"):
        net.allow("nodekey:x")
    net.close()


def test_close_kills_a_helper_whose_stdin_is_full(fake_helper, monkeypatch):
    monkeypatch.setenv("FAKE_LINK_MODE", "stuck")
    net = tailnet.Tailnet()
    net.start()
    proc = net._proc
    writer = threading.Thread(
        target = lambda: pytest.raises(
            tailnet.TailnetError, net._call, "allow", timeout = 1, pubkey = "x" * (4 << 20)
        ),
        daemon = True,
    )
    writer.start()
    time.sleep(0.5)
    closer = threading.Thread(target = net.close, daemon = True)
    closer.start()
    closer.join(20)
    assert not closer.is_alive() and proc.poll() is not None
    writer.join(10)
    assert not writer.is_alive()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "PDEATHSIG is Linux only")
@pytest.mark.parametrize("studio", [False, True], ids = ["cli", "studio"])
def test_the_helper_dies_with_its_parent_even_if_a_child_holds_its_stdin(tmp_path, studio):
    script = tmp_path / "fake_link.py"
    script.write_text(FAKE_HELPER)
    repo = Path(__file__).resolve().parents[2]
    backend = [str(repo / "studio" / "backend")] if studio else []
    parent = textwrap.dedent(
        f"""
        import os, sys, threading
        sys.path[:0] = [{str(repo)!r}] + {backend!r}
        if {studio}:
            import utils.process_lifetime
        from studio.cluster_link import tailnet
        tailnet._command = lambda: [sys.executable, {str(script)!r}]
        net = tailnet.Tailnet()
        starter = threading.Thread(target = net.start)  # a short-lived thread, like to_thread
        starter.start()
        starter.join()
        print(net._proc.pid, flush = True)
        if os.fork() == 0:
            os.close(1)
            os.close(2)
            import time
            time.sleep(60)  # holds the helper's stdin open
            os._exit(0)
        import time
        time.sleep(1)
        assert net._proc.poll() is None, "the helper died with the thread that started it"
        print("alive", flush = True)
        os._exit(0)
        """
    )
    env = dict(os.environ, FAKE_LINK_LOG = str(tmp_path / "link.log"))
    env.pop(tailnet.ENV_DISABLE, None)
    out = subprocess.run(
        [sys.executable, "-c", parent], env = env, capture_output = True, text = True, timeout = 60
    )
    assert out.stdout.split()[1:] == ["alive"], out.stderr[-2000:]
    pid = int(out.stdout.split()[0])
    deadline = time.monotonic() + 10
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    alive = _alive(pid)
    if alive:
        os.kill(pid, 9)
    assert not alive, "the helper outlived its parent"


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(") ", 1)[1][0] != "Z"
    except (FileNotFoundError, ProcessLookupError):  # reaped while being read
        return False


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

    state["installed"] = True
    assert remote.ensure_helper() and state["calls"] == 2
    monkeypatch.setattr(remote, "_HELPER", {})
    state.update(installed = False, fail = True)
    clock = [1000.0]
    monkeypatch.setattr(remote.time, "monotonic", lambda: clock[0])
    assert not remote.ensure_helper() and state["calls"] == 3
    clock[0] += remote.INSTALL_RETRY_S - 1
    assert not remote.ensure_helper() and state["calls"] == 3
    clock[0] += 2
    state["fail"] = False
    assert remote.ensure_helper() and state["calls"] == 4

    monkeypatch.setattr(remote, "_HELPER", {})
    monkeypatch.setenv("UNSLOTH_CLUSTER_TAILNET", "0")
    assert not remote.ensure_helper(notes.append) and state["calls"] == 4 and len(notes) == 1


class _Client:
    """A tailnet client recording forwards; ``gate`` holds a forward mid-call."""

    def __init__(
        self,
        log,
        gate = None,
    ):
        self.log, self.gate, self.dead, self.closed = log, gate, False, False

    def start(self):
        self.log.append(("start", self))

    def forward(
        self,
        address,
        port,
        *,
        key_path = None,
    ):
        if self.gate is not None:
            self.gate.wait(5)
        self.log.append(("forward", address, port))
        return 40000 + port

    def unforward(
        self,
        address,
        remote_port = None,
    ):
        self.log.append(("unforward", address))

    def close(self):
        self.closed = True


@pytest.fixture
def fake_remote(monkeypatch, tmp_path):
    from studio.cluster_link import remote

    log, made = [], []
    fake = type("TN", (), {})()
    fake.Tailnet = lambda: made.append(_Client(log)) or made[-1]
    monkeypatch.setattr(remote, "_tailnet", lambda: fake)
    monkeypatch.setattr(remote, "own_pub", lambda kind, **kw: "nodekey:c")
    monkeypatch.setattr(remote, "_CLIENT", None)
    monkeypatch.setattr(remote, "_FORWARDS", {})
    return remote, log, made


def test_a_drop_during_a_forward_is_not_undone(fake_remote):
    remote, log, _ = fake_remote
    gate = threading.Event()
    remote._CLIENT = _Client(log, gate)
    t = threading.Thread(target = remote.forward, args = ("n1", "tcA", 50070))
    t.start()
    time.sleep(0.2)
    dropper = threading.Thread(target = remote.drop, args = ("n1",))
    dropper.start()
    time.sleep(0.2)
    gate.set()
    t.join(5)
    dropper.join(5)
    assert remote.forwarded() == []
    assert log == [("forward", "tcA", 50070), ("unforward", "tcA")]


def test_a_changed_address_unforwards_the_old_one(fake_remote):
    remote, log, _ = fake_remote
    assert remote.forward("n1", "tcA", 50070) == 90070
    assert remote.forward("n1", "tcB", 50070) == 90070
    assert ("unforward", "tcA") in log and remote._FORWARDS["n1"] == ("tcB", {50070})
    remote.drop("n1")
    assert log[-1] == ("unforward", "tcB")


def test_a_dead_shared_client_is_replaced(fake_remote):
    remote, log, made = fake_remote
    assert remote.forward("n1", "tcA", 50070) == 90070
    first = made[0]
    first.dead = True
    assert remote.forward("n1", "tcA", 50070) == 90070
    assert len(made) == 2 and first.closed and remote._CLIENT is made[1]


def test_status_checks_drop_forwards_of_removed_workers(fake_remote):
    from studio.cluster_link import status

    remote, log, _ = fake_remote
    remote.forward("gone", "tcA", 50070)
    remote.forward("live", "tcB", 50070)
    status._drop_removed({"live"})
    assert remote.forwarded() == ["live"] and ("unforward", "tcA") in log


def test_key_creation_and_leave_share_the_identity_lock(tmp_path, monkeypatch):
    from studio.cluster_link import identity, remote

    monkeypatch.setenv("UNSLOTH_CLUSTER_STATE_DIR", str(tmp_path / "state"))
    held = []
    fake = type("TN", (), {})()

    def genkey(path, **_kw):
        # Another process holding the lock would block here: probe it without waiting.
        fd = os.open(str(tmp_path / "state" / "identity.lock"), os.O_RDWR)
        try:
            identity.lock_fd(fd, blocking = False)
            held.append(False)
        except OSError:
            held.append(True)
        finally:
            identity.unlock_fd(fd)
        Path(path).write_text("{}")
        return "nodekey:" + "c" * 64

    fake.new_client_key = genkey
    monkeypatch.setattr(remote, "_tailnet", lambda: fake)
    assert remote.own_pub("client") == "nodekey:" + "c" * 64
    assert held == [True]
    remote.key_path("client").with_suffix(".pub").unlink()
    assert remote.own_pub("client") == "nodekey:" + "c" * 64 and held == [True, True]
    assert remote.key_path("client").with_suffix(".pub").read_text() == "nodekey:" + "c" * 64


def test_real_genkey_keeps_a_key_whose_sidecar_was_lost(tmp_path, monkeypatch):
    from studio.cluster_link import remote

    _real_helper()
    monkeypatch.delenv(tailnet.ENV_DISABLE, raising = False)
    monkeypatch.setenv("UNSLOTH_CLUSTER_STATE_DIR", str(tmp_path / "state"))
    pub = remote.own_pub("client")
    key = remote.key_path("client")
    body = key.read_bytes()
    key.with_suffix(".pub").unlink()
    assert remote.own_pub("client") == pub and key.read_bytes() == body
    assert key.with_suffix(".pub").read_text() == pub
    assert tailnet.new_client_key(key) == pub
    if os.name != "nt":
        assert (key.stat().st_mode & 0o777) == 0o600
    assert [p.name for p in key.parent.iterdir() if p.name.startswith(".")] == []


def test_real_helper_exits_once_reparented_though_stdin_stays_open(tmp_path):
    helper = _real_helper()
    if os.name == "nt":
        pytest.skip("Windows has no reparenting; the Job Object and stdin EOF cover it")
    parent = textwrap.dedent(
        f"""
        import os, subprocess, sys, time
        proc = subprocess.Popen([{str(helper)!r}, "stdio", "--parent-pid", str(os.getpid())],
                                stdin = subprocess.PIPE, stdout = subprocess.PIPE)
        assert b'"ready"' in proc.stdout.readline()
        print(proc.pid, flush = True)
        if os.fork() == 0:
            os.close(1)
            os.close(2)
            time.sleep(60)  # keeps the helper's stdin open
            os._exit(0)
        os._exit(0)
        """
    )
    out = subprocess.run([sys.executable, "-c", parent], capture_output = True, text = True, timeout = 60)
    assert out.stdout.strip(), out.stderr[-2000:]
    pid = int(out.stdout.split()[0])
    deadline = time.monotonic() + 10
    while _gone(pid) is False and time.monotonic() < deadline:
        time.sleep(0.1)
    gone = _gone(pid)
    if not gone:
        os.kill(pid, 9)
    assert gone, "the helper outlived its parent"


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return sys.platform.startswith("linux") and not _alive(pid)
