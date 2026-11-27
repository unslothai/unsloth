# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sandboxed curl/wget reach the network only through a per-call proxy that refuses metadata and local addresses."""

from __future__ import annotations

import http.server
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

import pytest

if sys.platform == "win32":
    pytest.skip("the egress proxy backs the POSIX sandboxes", allow_module_level = True)

from core.inference import os_sandbox, sandbox_egress, tools

_SESSION = "__LOCALID_sandbox_egress"
_METADATA_URL = "http://169.254.169.254/latest/meta-data/"


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/redirect"):
            self.send_response(302)
            self.send_header("Location", _METADATA_URL)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = b"hello from the host"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def origin():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture
def loopback_origin_allowed(monkeypatch):
    # The test origin listens on loopback, which the proxy refuses; let only that through.
    real = sandbox_egress.refused_address
    monkeypatch.setattr(
        sandbox_egress,
        "refused_address",
        lambda address: address != "127.0.0.1" and real(address),
    )


@pytest.fixture
def proxy(monkeypatch):
    # conftest's network guard lists loopback in no_proxy, which would send these requests around the proxy.
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")
    running = sandbox_egress.EgressProxy()
    yield f"http://127.0.0.1:{running.port}"
    running.close()


class _AlwaysProxy(urllib.request.ProxyHandler):
    # urllib skips the proxy for macOS's system bypass list, which holds 169.254/16; curl does not.
    def proxy_open(self, req, proxy, type):
        req.set_proxy(proxy.split("//", 1)[1], "http")


def _get(url: str, proxy_url: str) -> tuple[int, str]:
    opener = urllib.request.build_opener(_AlwaysProxy({"http": proxy_url}))
    try:
        with opener.open(url, timeout = 10) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def _connect(proxy_url: str, target: str) -> bytes:
    port = int(proxy_url.rsplit(":", 1)[1])
    with socket.create_connection(("127.0.0.1", port), timeout = 10) as sock:
        sock.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
        return sock.recv(4096)


@pytest.mark.parametrize(
    "address",
    [
        "169.254.169.254",
        "169.254.10.10",
        "fe80::1",
        "fe80::1%eth0",
        "::ffff:169.254.169.254",
        "2002:a9fe:a9fe::",
        "64:ff9b::a9fe:a9fe",
        "64:ff9b:1::a9fe:a9fe",
        "64:ff9b:1::7f00:1",
        "fd00:ec2::254",
        "100.100.100.200",
        "127.0.0.1",
        "127.8.8.8",
        "::1",
        "0.0.0.0",
        "0.1.2.3",
        "::",
        "224.0.0.1",
    ],
)
def test_metadata_link_local_and_loopback_addresses_are_refused(address):
    assert sandbox_egress.refused_address(address)


@pytest.mark.parametrize("address", ["93.184.215.14", "10.0.0.5", "192.168.1.2", "2606:4700::1111"])
def test_public_and_lan_addresses_are_allowed(address):
    assert not sandbox_egress.refused_address(address)


@pytest.mark.parametrize(
    "host",
    ["169.254.169.254", "metadata.google.internal", "metadata", "METADATA.GOOGLE.INTERNAL."],
)
def test_metadata_hosts_are_refused_before_any_lookup(monkeypatch, host):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: pytest.fail("looked up"))
    with pytest.raises(sandbox_egress.EgressRefused):
        sandbox_egress.resolve_allowed(host, 80)


def test_a_name_resolving_to_metadata_is_refused(monkeypatch):
    answers = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 80)),
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 80)),
    ]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: answers)
    with pytest.raises(sandbox_egress.EgressRefused, match = "169.254.169.254"):
        sandbox_egress.resolve_allowed("rebind.example", 80)


def test_the_proxy_dials_the_address_it_checked(
    monkeypatch, proxy, origin, loopback_origin_allowed
):
    port = int(origin.rsplit(":", 1)[1])
    real = socket.getaddrinfo
    answers = iter(["127.0.0.1", "169.254.169.254"])

    def rebinding(host, *args, **kwargs):
        if host != "rebind.test":
            return real(host, *args, **kwargs)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (next(answers), port))]

    monkeypatch.setattr(socket, "getaddrinfo", rebinding)
    assert _get(f"http://rebind.test:{port}/", proxy) == (200, "hello from the host")
    # The second answer would have been the rebind; it was never asked for.
    assert next(answers) == "169.254.169.254"


def test_the_proxy_refuses_metadata_over_http_and_connect(proxy):
    status, body = _get(_METADATA_URL, proxy)
    assert status == 403 and "cloud-metadata" in body
    assert _connect(proxy, "169.254.169.254:443").startswith(b"HTTP/1.1 403")
    assert _connect(proxy, "metadata.google.internal:80").startswith(b"HTTP/1.1 403")
    assert _connect(proxy, "[fe80::1]:443").startswith(b"HTTP/1.1 403")


def test_the_proxy_refuses_the_host_loopback(proxy, origin):
    status, body = _get(origin, proxy)
    assert status == 403 and "127.0.0.1" in body


def test_the_proxy_forwards_an_allowed_host(proxy, origin, loopback_origin_allowed):
    assert _get(origin, proxy) == (200, "hello from the host")
    assert _connect(proxy, origin.split("//", 1)[1]).startswith(b"HTTP/1.1 200")


def test_a_redirect_to_metadata_is_refused(proxy, origin, loopback_origin_allowed):
    status, body = _get(f"{origin}/redirect", proxy)
    assert status == 403 and "169.254.169.254" in body


def test_the_proxy_serves_a_unix_socket_and_stops_on_close(origin, loopback_origin_allowed):
    # macOS caps AF_UNIX paths at 104 bytes, which its pytest tmp_path exceeds.
    directory = tempfile.mkdtemp(prefix = "us-egress-", dir = "/tmp")
    path = os.path.join(directory, "proxy.sock")
    running = sandbox_egress.EgressProxy(path)
    port = origin.rsplit(":", 1)[1]
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(10)
        sock.connect(path)
        sock.sendall(f"GET {origin}/ HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
        reply = b""
        while chunk := sock.recv(4096):
            reply += chunk
    assert reply.startswith(b"HTTP/1.") and reply.endswith(b"hello from the host")
    running.close()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock, pytest.raises(OSError):
        sock.connect(path)
    shutil.rmtree(directory, ignore_errors = True)


def test_open_tunnels_end_with_the_proxy(origin, loopback_origin_allowed):
    before = set(threading.enumerate())
    running = sandbox_egress.EgressProxy()
    target = origin.split("//", 1)[1]
    tunnels = []
    for _ in range(20):
        sock = socket.create_connection(("127.0.0.1", running.port), timeout = 10)
        sock.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
        assert sock.recv(4096).startswith(b"HTTP/1.1 200")
        tunnels.append(sock)
    running.close()
    deadline = time.monotonic() + 5
    while set(threading.enumerate()) - before and time.monotonic() < deadline:
        time.sleep(0.05)
    for sock in tunnels:
        sock.close()
    assert not set(threading.enumerate()) - before


def test_seatbelt_reaches_only_its_proxy(tmp_path, monkeypatch):
    from core.inference import sandbox_macos

    monkeypatch.setattr(sandbox_macos, "available", lambda: (True, "stubbed for this test"))
    workdir = tmp_path / "session"
    workdir.mkdir()
    plain = sandbox_macos.prepare(
        os_sandbox.ToolLaunchPlan(argv = ("/bin/echo",), workdir = str(workdir), env = {})
    )
    proxied = sandbox_macos.prepare(
        os_sandbox.ToolLaunchPlan(
            argv = ("/bin/echo",),
            workdir = str(workdir),
            env = {"no_proxy": "*", "NO_PROXY": "*"},
            egress_proxy = True,
        )
    )
    try:
        port = int(proxied.egress_proxy.rsplit(":", 1)[1])
        assert plain.egress_proxy is None and "http_proxy" not in plain.env
        assert '(allow network-outbound (remote ip "*:*"))' in plain.argv[2]
        assert '(remote ip "*:*")' not in proxied.argv[2]
        assert f'(allow network-outbound (remote ip "localhost:{port}"))' in proxied.argv[2]
        assert proxied.env["https_proxy"] == proxied.env["HTTP_PROXY"] == proxied.egress_proxy
        assert proxied.env["no_proxy"] == proxied.env["NO_PROXY"] == sandbox_egress.NO_PROXY
        rc_dir = proxied.env["CURL_HOME"]
        assert os.path.getsize(os.path.join(rc_dir, ".curlrc")) == 0
        assert (
            proxied.env["WGETRC"] == proxied.env["SYSTEM_WGETRC"] == os.path.join(rc_dir, "wgetrc")
        )
        with open(proxied.env["WGETRC"]) as f:
            assert f.read() == "netrc = off\n"
        assert rc_dir in proxied.argv[2] and "CURL_HOME" not in plain.env
        socket.create_connection(("127.0.0.1", port), timeout = 5).close()
    finally:
        plain.cleanup()
        proxied.cleanup()
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout = 5)
    assert not os.path.exists(rc_dir)


def _plans(monkeypatch) -> list:
    plans = []

    def record(plan, **kwargs):
        plans.append(plan)
        raise os_sandbox.SandboxBuildError("recorded")

    monkeypatch.setattr(tools, "_prepare_tool_launch", record)
    return plans


@pytest.mark.parametrize(
    "command,proxied",
    [
        ("curl -sS https://example.com", True),
        ("wget -q https://example.com", True),
        ("chmod +x a && ./a", False),
        ("sleep 5 & kill $!", False),
        ("echo hi", False),
        ("python -c 'print(1)'", False),
    ],
)
def test_only_the_downloader_exemption_asks_for_the_proxy(monkeypatch, command, proxied):
    # Runners may lack wget on the sandbox PATH; that check has its own tests.
    monkeypatch.setattr(tools, "_on_safe_path", lambda name: True)
    plans = _plans(monkeypatch)
    tools._bash_exec(command, None, 60, _SESSION)
    assert [plan.egress_proxy for plan in plans] == [proxied]


def _isolated() -> bool:
    if sys.platform != "linux" or not os_sandbox.capability_snapshot().available:
        return False
    probe = subprocess.run(
        [
            "bwrap",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-net",
            "--ro-bind",
            "/",
            "/",
            "true",
        ],
        capture_output = True,
    )
    return probe.returncode == 0


needs_bwrap = pytest.mark.skipif(not _isolated(), reason = "needs bubblewrap with a private network")


@needs_bwrap
def test_bubblewrap_unshares_the_network_only_for_proxied_calls(tmp_path):
    from core.inference import sandbox_linux

    plain = sandbox_linux.prepare(
        os_sandbox.ToolLaunchPlan(argv = ("/bin/true",), workdir = str(tmp_path), env = {})
    )
    proxied = sandbox_linux.prepare(
        os_sandbox.ToolLaunchPlan(
            argv = ("/bin/true",), workdir = str(tmp_path), env = {}, egress_proxy = True
        )
    )
    try:
        assert "--unshare-net" not in plain.argv and plain.egress_proxy is None
        assert "--unshare-net" in proxied.argv
        command = proxied.argv[proxied.argv.index("--") + 1 :]
        assert command[1:4] == ("-I", "-S", "/run/unsloth-egress/relay.py")
        assert command[-1] == "/bin/true"
        assert proxied.egress_proxy == "http://127.0.0.1:3128"
    finally:
        plain.cleanup()
        proxied.cleanup()


@needs_bwrap
def test_a_launch_without_the_proxy_keeps_downloaders_blocked(monkeypatch):
    from core.inference import sandbox_linux

    monkeypatch.setattr(sandbox_linux, "_private_network_works", lambda *a: False)
    assert tools._bash_exec("curl --version", None, 60, _SESSION) == (
        "Blocked command(s) for safety: curl"
    )
    assert tools._bash_exec("wget --version; chmod +x a", None, 60, _SESSION) == (
        "Blocked command(s) for safety: wget"
    )
    assert "Blocked" not in tools._bash_exec("touch a; chmod +x a", None, 60, _SESSION)


@needs_bwrap
def test_a_proxy_that_cannot_start_keeps_downloaders_blocked(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("no socket for you")

    monkeypatch.setattr(sandbox_egress, "EgressProxy", refuse)
    assert tools._bash_exec("curl --version", None, 60, _SESSION) == (
        "Blocked command(s) for safety: curl"
    )


def _bash(command: str) -> str:
    if shutil.which("curl") is None:
        pytest.skip("curl is not installed")
    return tools._bash_exec(command, None, 60, _SESSION)


@needs_bwrap
def test_sandboxed_curl_reaches_an_allowed_host_only_through_the_proxy(
    origin, loopback_origin_allowed
):
    # --noproxy '' drops the loopback exemption so the request goes to the host through the proxy.
    out = _bash(f"curl -sS --noproxy '' {origin}/")
    assert out == "hello from the host", out
    direct = _bash(f"curl -sS -m 5 --noproxy '*' {origin}/; echo rc=$?")
    assert "hello" not in direct and "rc=7" in direct, direct


@needs_bwrap
def test_sandboxed_curl_cannot_reach_metadata(origin, loopback_origin_allowed):
    out = _bash(f"curl -sS {_METADATA_URL}")
    assert "refused" in out and "cloud-metadata" in out, out
    redirected = _bash(f"curl -sS -L --noproxy '' {origin}/redirect")
    assert "refused" in redirected and "169.254.169.254" in redirected, redirected
    raw = _bash("curl -sS -m 5 --noproxy '*' http://169.254.169.254/; echo rc=$?")
    assert "rc=7" in raw or "rc=28" in raw, raw


@needs_bwrap
def test_a_workdir_curlrc_is_not_read(origin, loopback_origin_allowed):
    workdir = tools._get_workdir(_SESSION)
    rc = os.path.join(workdir, ".curlrc")
    with open(rc, "w") as f:
        f.write('write-out = "RC-APPLIED"\n')
    try:
        out = _bash(f"curl -sS --noproxy '' {origin}/")
    finally:
        os.remove(rc)
    assert out == "hello from the host", out


@needs_bwrap
def test_the_proxy_stops_when_the_call_ends(monkeypatch):
    started = []
    real = sandbox_egress.EgressProxy

    def tracked(*args, **kwargs):
        started.append(real(*args, **kwargs))
        return started[-1]

    monkeypatch.setattr(sandbox_egress, "EgressProxy", tracked)
    _bash("curl --version")
    assert len(started) == 1 and started[0]._closed and not started[0]._thread.is_alive()


@pytest.mark.skipif(shutil.which("wget") is None, reason = "wget is not installed")
def test_a_global_wgetrc_cannot_turn_a_download_into_an_upload(
    tmp_path, origin, loopback_origin_allowed
):
    hostile = tmp_path / "hostile"
    hostile.write_text("post_data = LEAKED\n", encoding = "utf-8")
    env = dict(os.environ, SYSTEM_WGETRC = str(hostile))
    env.update(sandbox_egress.proxy_environment("http://127.0.0.1:9", str(tmp_path)))
    sandbox_egress.write_empty_rc_files(str(tmp_path))
    for name in (
        "http_proxy",
        "HTTP_PROXY",
        "https_proxy",
        "HTTPS_PROXY",
        "all_proxy",
        "ALL_PROXY",
    ):
        env.pop(name)
    out = subprocess.run(
        ["wget", "-qO-", "--no-proxy", f"{origin}/"], env = env, capture_output = True, text = True
    )
    assert out.stdout == "hello from the host", out


@pytest.mark.parametrize("unix", [False, True])
def test_closing_an_idle_proxy_does_not_wait_out_the_accept_timeout(tmp_path, monkeypatch, unix):
    # Relative socket path: a deep tmp_path can pass the AF_UNIX length limit.
    monkeypatch.chdir(tmp_path)
    proxy = sandbox_egress.EgressProxy("p.sock" if unix else None)
    time.sleep(0.05)  # let the accept loop block
    started = time.perf_counter()
    proxy.close()
    elapsed = time.perf_counter() - started
    assert not proxy._thread.is_alive()
    # Every proxied curl/wget call paid up to the 0.25 s accept timeout here.
    assert elapsed < 0.15, elapsed


def test_a_second_close_sends_no_second_wake_up(monkeypatch):
    proxy = sandbox_egress.EgressProxy()
    wakes = []
    real = proxy._wake_listener
    monkeypatch.setattr(proxy, "_wake_listener", lambda: wakes.append(1) or real())
    proxy.close()
    proxy.close()
    assert wakes == [1]
