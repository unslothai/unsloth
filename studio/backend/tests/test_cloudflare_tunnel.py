# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the Cloudflare quick-tunnel helper and run.py wiring.

cloudflare_tunnel.py is stdlib-only (storage_roots is imported lazily), so it
loads via spec_from_file_location without the studio venv. run.py defaults are
checked by AST so we never import its heavy deps (uvicorn/structlog).
"""

import ast
import importlib.util
import io
import os
import sys
import tarfile
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_CT_PY = _BACKEND / "cloudflare_tunnel.py"
_RUN_PY = _BACKEND / "run.py"


def _load_ct():
    spec = importlib.util.spec_from_file_location("cloudflare_tunnel", _CT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ct = _load_ct()


# ── URL parsing ──────────────────────────────────────────────────────


def test_url_regex_extracts_and_ignores_noise():
    blob = (
        "2026-06-11T10:00:00Z INF Thank you for trying Cloudflare Tunnel.\n"
        "2026-06-11T10:00:01Z INF Requesting new quick Tunnel on trycloudflare.com...\n"
        "2026-06-11T10:00:01Z INF |  https://setting-democracy-gathering.trycloudflare.com  |\n"
        "2026-06-11T10:00:02Z INF Registered tunnel connection https://not-the-url.example.com\n"
    )
    m = ct._URL_RE.search(blob)
    assert m is not None
    assert m.group(0) == "https://setting-democracy-gathering.trycloudflare.com"


def test_url_regex_no_match_on_unrelated():
    assert ct._URL_RE.search("INF connecting to https://api.cloudflare.com/v4") is None


def test_url_regex_ignores_api_endpoint():
    # cloudflared's failure line names its own API host; it must never be taken
    # as the tunnel URL (it returns a 404 and is not a quick tunnel).
    line = (
        'failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel": '
        "context deadline exceeded"
    )
    assert ct._URL_RE.search(line) is None


def test_url_regex_skips_api_host_but_matches_real_url():
    blob = (
        'ERR failed to request quick Tunnel: Post "https://api.trycloudflare.com/tunnel"\n'
        "INF |  https://brave-mountain-river-clouds.trycloudflare.com  |\n"
    )
    m = ct._URL_RE.search(blob)
    assert m is not None
    assert m.group(0) == "https://brave-mountain-river-clouds.trycloudflare.com"


# ── asset mapping ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "system,machine,expected",
    [
        ("Linux", "x86_64", ("cloudflared-linux-amd64", False)),
        ("Linux", "aarch64", ("cloudflared-linux-arm64", False)),
        ("Darwin", "arm64", ("cloudflared-darwin-arm64.tgz", True)),
        ("Darwin", "x86_64", ("cloudflared-darwin-amd64.tgz", True)),
        ("Windows", "AMD64", ("cloudflared-windows-amd64.exe", False)),
        ("Windows", "x86", ("cloudflared-windows-386.exe", False)),
        ("Linux", "mips", None),
        ("Plan9", "x86_64", None),
    ],
)
def test_asset_name(monkeypatch, system, machine, expected):
    monkeypatch.setattr(ct.platform, "system", lambda: system)
    monkeypatch.setattr(ct.platform, "machine", lambda: machine)
    assert ct._asset_name() == expected


# ── binary discovery ─────────────────────────────────────────────────


def test_find_cloudflared_prefers_path(monkeypatch):
    monkeypatch.setattr(ct.shutil, "which", lambda name: "/usr/local/bin/cloudflared")
    assert ct.find_cloudflared() == "/usr/local/bin/cloudflared"


def test_find_cloudflared_falls_back_to_cache(monkeypatch, tmp_path):
    cached = tmp_path / "cloudflared"
    cached.write_text("#!/bin/sh\n")
    cached.chmod(0o755)
    monkeypatch.setattr(ct.shutil, "which", lambda name: None)
    monkeypatch.setattr(ct, "_cache_path", lambda: cached)
    assert ct.find_cloudflared() == str(cached)


def test_find_cloudflared_none_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(ct.shutil, "which", lambda name: None)
    monkeypatch.setattr(ct, "_cache_path", lambda: tmp_path / "absent")
    assert ct.find_cloudflared() is None


# ── ensure / download ────────────────────────────────────────────────


def test_ensure_downloads_and_chmods_when_missing(monkeypatch, tmp_path):
    cached = tmp_path / "cloudflared"
    monkeypatch.setattr(ct, "find_cloudflared", lambda: None)
    monkeypatch.setattr(ct, "_asset_name", lambda: ("cloudflared-linux-amd64", False))
    monkeypatch.setattr(ct, "_cache_path", lambda: cached)

    def fake_download(url, dest):
        assert url.endswith("/cloudflared-linux-amd64")
        dest.write_bytes(b"ELF-ish")
        return True

    monkeypatch.setattr(ct, "_download", fake_download)
    monkeypatch.setattr(ct.sys, "platform", "linux")
    path = ct.ensure_cloudflared()
    assert path == str(cached)
    assert cached.exists()
    # Host OS, not monkeypatched ct.sys.platform.
    if os.name != "nt":
        assert cached.stat().st_mode & 0o111


def test_ensure_returns_none_on_download_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(ct, "find_cloudflared", lambda: None)
    monkeypatch.setattr(ct, "_asset_name", lambda: ("cloudflared-linux-amd64", False))
    monkeypatch.setattr(ct, "_cache_path", lambda: tmp_path / "cloudflared")
    monkeypatch.setattr(ct, "_download", lambda url, dest: False)
    assert ct.ensure_cloudflared() is None


def test_ensure_returns_none_for_unsupported_arch(monkeypatch, tmp_path):
    monkeypatch.setattr(ct, "find_cloudflared", lambda: None)
    monkeypatch.setattr(ct, "_asset_name", lambda: None)
    monkeypatch.setattr(ct, "_cache_path", lambda: tmp_path / "cloudflared")
    assert ct.ensure_cloudflared() is None


def test_download_sets_user_agent(monkeypatch, tmp_path):
    import urllib.request

    captured = {}

    class _Resp:
        _sent = False

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n = -1):
            if self._sent:
                return b""
            self._sent = True
            return b"payload"

    def fake_urlopen(req, timeout = None):
        captured["ua"] = req.get_header("User-agent")
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "cloudflared"
    assert ct._download("https://github.com/cloudflare/cloudflared/x", dest) is True
    assert captured["ua"] == "unsloth-studio"  # GitHub CDN 403s the default UA
    assert dest.read_bytes() == b"payload"


# ── cross-platform: Windows (.exe), macOS (.tgz) ─────────────────────


def test_cache_path_uses_exe_on_windows(monkeypatch, tmp_path):
    import types

    fake_sr = types.ModuleType("utils.paths.storage_roots")
    fake_sr.studio_bin_root = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "utils.paths.storage_roots", fake_sr)
    monkeypatch.setattr(ct.sys, "platform", "win32")
    assert ct._cache_path() == tmp_path / "cloudflared.exe"


def test_ensure_windows_downloads_exe(monkeypatch, tmp_path):
    cached = tmp_path / "cloudflared.exe"
    monkeypatch.setattr(ct, "find_cloudflared", lambda: None)
    monkeypatch.setattr(ct, "_asset_name", lambda: ("cloudflared-windows-amd64.exe", False))
    monkeypatch.setattr(ct, "_cache_path", lambda: cached)
    monkeypatch.setattr(ct.sys, "platform", "win32")

    def fake_download(url, dest):
        assert url.endswith("/cloudflared-windows-amd64.exe")
        dest.write_bytes(b"MZ")  # PE header magic
        return True

    monkeypatch.setattr(ct, "_download", fake_download)
    # chmod is skipped on Windows; would raise on a path that does not exist yet.
    monkeypatch.setattr(ct.os, "chmod", lambda *a, **k: pytest.fail("chmod called on win32"))
    assert ct.ensure_cloudflared() == str(cached)
    assert cached.read_bytes() == b"MZ"


def test_ensure_macos_extracts_tgz_and_chmods(monkeypatch, tmp_path):
    cached = tmp_path / "cloudflared"
    monkeypatch.setattr(ct, "find_cloudflared", lambda: None)
    monkeypatch.setattr(ct, "_asset_name", lambda: ("cloudflared-darwin-arm64.tgz", True))
    monkeypatch.setattr(ct, "_cache_path", lambda: cached)
    monkeypatch.setattr(ct.sys, "platform", "darwin")

    def fake_download(url, dest):
        # dest is cached.with_suffix(".tgz"); write a real archive there.
        assert url.endswith("/cloudflared-darwin-arm64.tgz")
        with tarfile.open(dest, "w:gz") as tar:
            data = b"mach-o"
            info = tarfile.TarInfo(name = "cloudflared")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        return True

    monkeypatch.setattr(ct, "_download", fake_download)
    path = ct.ensure_cloudflared()
    assert path == str(cached)
    assert cached.read_bytes() == b"mach-o"
    if os.name != "nt":
        assert cached.stat().st_mode & 0o111
    assert not cached.with_suffix(".tgz").exists()  # temp archive cleaned up


# ── .tgz extraction (darwin) ─────────────────────────────────────────


def _make_tgz(
    tmp_path,
    member_name,
    data = b"bin",
):
    tgz = tmp_path / "cf.tgz"
    with tarfile.open(tgz, "w:gz") as tar:
        info = tarfile.TarInfo(name = member_name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return tgz


def test_tgz_extraction_extracts_clean_member(tmp_path):
    tgz = _make_tgz(tmp_path, "cloudflared")
    dest = tmp_path / "out"
    assert ct._extract_tgz_member(tgz, dest) is True
    assert dest.read_bytes() == b"bin"


def test_tgz_extraction_rejects_traversal(tmp_path):
    tgz = _make_tgz(tmp_path, "../cloudflared")
    dest = tmp_path / "out"
    assert ct._extract_tgz_member(tgz, dest) is False
    assert not dest.exists()


def test_tgz_extraction_missing_member(tmp_path):
    tgz = _make_tgz(tmp_path, "README")
    dest = tmp_path / "out"
    assert ct._extract_tgz_member(tgz, dest) is False


# ── tunnel lifecycle ─────────────────────────────────────────────────


class _FakePopen:
    def __init__(self):
        self.terminated = False
        self.killed = False
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated = True
        self._alive = False

    def wait(self, timeout = None):
        if self._alive:
            raise ct.subprocess.TimeoutExpired(cmd = "cloudflared", timeout = timeout)
        return 0

    def kill(self):
        self.killed = True
        self._alive = False


def test_stop_terminates_process():
    t = ct.CloudflareTunnel(8080, "/bin/cloudflared")
    fake = _FakePopen()
    t._proc = fake
    t.stop()
    assert fake.terminated is True
    assert t._proc is None
    # second stop is a no-op (idempotent)
    t.stop()


def test_start_after_stop_does_not_spawn(monkeypatch):
    # If stop() lands before start() (a concurrent shutdown in the caller's
    # register->start window), start() must NOT spawn a cloudflared process --
    # nobody would own it and it would be orphaned.
    t = ct.CloudflareTunnel(8080, "/bin/cloudflared")
    spawned = []

    class _FakeProc:
        stdout = None

        def poll(self):
            return 0

    monkeypatch.setattr(ct.subprocess, "Popen", lambda *a, **k: (spawned.append(a), _FakeProc())[1])
    t.stop()  # proc is None -> no-op terminate, but marks the tunnel stopped
    t.start()  # must short-circuit before Popen
    assert spawned == []
    assert t._proc is None


def test_wait_for_ready_times_out_without_blocking():
    t = ct.CloudflareTunnel(8080, "/bin/cloudflared")
    assert t.wait_for_ready(timeout = 0.05) is None


def _fake_proc(text):
    return types.SimpleNamespace(stdout = io.StringIO(text))


def test_reader_captures_url_and_registration():
    t = ct.CloudflareTunnel(8080, "/bin/cloudflared")
    t._reader(
        _fake_proc(
            "INF Requesting new quick Tunnel on trycloudflare.com...\n"
            "INF |  https://words-here-abc.trycloudflare.com  |\n"
            "INF Registered tunnel connection connIndex=0 protocol=http2\n"
        )
    )
    assert t.url == "https://words-here-abc.trycloudflare.com"
    assert t.ready is True
    assert t.wait_for_ready(0) == t.url
    assert t.error is None  # a fully-registered tunnel records no error


def test_reader_url_without_registration_is_not_ready():
    # A URL but no "Registered tunnel connection" (e.g. quic control stream
    # fails) must not be advertised -- it returns Cloudflare error 1033.
    t = ct.CloudflareTunnel(8080, "/bin/cloudflared")
    t._reader(
        _fake_proc(
            "INF |  https://words-here-abc.trycloudflare.com  |\n"
            'ERR failed to serve tunnel connection error="control stream failure"\n'
        )
    )
    assert t.url == "https://words-here-abc.trycloudflare.com"
    assert t.ready is False
    assert t.wait_for_ready(0) is None
    assert t.error == "cloudflared exited before the tunnel connection registered"


def test_reader_handles_none_stdout():
    # Popen.stdout can be None; _reader must not crash and must leave the tunnel
    # un-ready so wait_for_ready returns None.
    t = ct.CloudflareTunnel(8080, "/bin/cloudflared")
    t._reader(types.SimpleNamespace(stdout = None))
    assert t.url is None
    assert t.ready is False
    assert t.wait_for_ready(0) is None
    assert t.error == "cloudflared exited before emitting a tunnel URL"


def test_reader_ignores_api_endpoint_failure_line():
    t = ct.CloudflareTunnel(8080, "/bin/cloudflared")
    t._reader(
        _fake_proc(
            "ERR failed to request quick Tunnel: Post "
            '"https://api.trycloudflare.com/tunnel": context deadline exceeded\n'
        )
    )
    assert t.url is None
    assert t.wait_for_ready(0) is None
    assert t.error == "cloudflared exited before emitting a tunnel URL"


# ── public reachability probe ────────────────────────────────────────


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self, size = -1):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch, handler):
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout = None: handler(req))


@pytest.fixture(autouse = True)
def _stub_dns_wait(monkeypatch, request):
    if request.node.name.startswith("test_verify_public_url"):
        monkeypatch.setattr(ct, "_wait_for_dns", lambda *a, **kw: None)


def test_wait_for_dns_polls_until_answer(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req.full_url)
        if len(calls) < 3:
            return _FakeResponse(b'{"Status":3}')
        return _FakeResponse(b'{"Status":0,"Answer":[{"data":"104.16.0.1"}]}')

    _patch_urlopen(monkeypatch, handler)
    monkeypatch.setattr(ct.time, "sleep", lambda _s: None)
    ct._wait_for_dns("words.trycloudflare.com", ct.time.monotonic() + 5)
    assert len(calls) == 3
    assert "name=words.trycloudflare.com" in calls[0]


def test_wait_for_dns_gives_up_at_deadline(monkeypatch):
    _patch_urlopen(monkeypatch, lambda req: _FakeResponse(b'{"Status":3}'))
    monkeypatch.setattr(ct.time, "sleep", lambda _s: None)
    ct._wait_for_dns("words.trycloudflare.com", ct.time.monotonic() + 0.05)


def test_wait_for_dns_bails_on_doh_error(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req.full_url)
        raise OSError("blocked")

    _patch_urlopen(monkeypatch, handler)
    ct._wait_for_dns("words.trycloudflare.com", ct.time.monotonic() + 5)
    assert len(calls) == 1


def test_verify_public_url_accepts_studio_marker(monkeypatch):
    seen = {}

    def handler(req):
        seen["url"] = req.full_url
        return _FakeResponse(b'{"status":"healthy","service":"Unsloth UI Backend"}')

    _patch_urlopen(monkeypatch, handler)
    assert ct.verify_public_url("https://words.trycloudflare.com") is True
    assert seen["url"] == "https://words.trycloudflare.com/api/health"


def test_verify_public_url_waits_for_dns_first(monkeypatch):
    order = []
    monkeypatch.setattr(ct, "_wait_for_dns", lambda host, deadline: order.append(("dns", host)))

    def handler(req):
        order.append(("probe", req.full_url))
        return _FakeResponse(b'{"service":"Unsloth UI Backend"}')

    _patch_urlopen(monkeypatch, handler)
    assert ct.verify_public_url("https://words.trycloudflare.com") is True
    assert order[0] == ("dns", "words.trycloudflare.com")
    assert order[1][0] == "probe"


def test_verify_public_url_dns_wait_and_probe_share_deadline(monkeypatch):
    # An exhausted DNS wait leaves the probe a single attempt, not a fresh window.
    calls = []
    monkeypatch.setattr(ct, "_wait_for_dns", lambda host, deadline: None)

    def handler(req):
        calls.append(req.full_url)
        raise OSError("unreachable")

    _patch_urlopen(monkeypatch, handler)
    assert ct.verify_public_url("https://words.trycloudflare.com", timeout = 0) is False
    assert len(calls) == 1


def test_verify_public_url_retries_then_succeeds(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req.full_url)
        if len(calls) < 3:
            raise OSError("Name or service not known")
        return _FakeResponse(b'{"service":"Unsloth UI Backend"}')

    _patch_urlopen(monkeypatch, handler)
    monkeypatch.setattr(ct.time, "sleep", lambda _s: None)
    assert ct.verify_public_url("https://words.trycloudflare.com") is True
    assert len(calls) == 3


def test_verify_public_url_rejects_unreachable_host(monkeypatch):
    def handler(req):
        raise OSError("Name or service not known")

    _patch_urlopen(monkeypatch, handler)
    monkeypatch.setattr(ct.time, "sleep", lambda _s: None)
    assert ct.verify_public_url("https://words.trycloudflare.com", timeout = 0.05) is False


def test_verify_public_url_rejects_foreign_responder(monkeypatch):
    # e.g. a Cloudflare error page: no service marker in the body.
    _patch_urlopen(monkeypatch, lambda req: _FakeResponse(b"<html>error 1033</html>"))
    monkeypatch.setattr(ct.time, "sleep", lambda _s: None)
    assert ct.verify_public_url("https://words.trycloudflare.com", timeout = 0.05) is False


@pytest.fixture(autouse = True)
def _stub_public_probe(monkeypatch, request):
    # start_studio_tunnel tests use fake hostnames; keep them off the network.
    if not request.node.name.startswith("test_start_studio_tunnel"):
        return
    monkeypatch.setattr(ct, "verify_public_url", lambda url, **kw: True)


def test_start_studio_tunnel_no_binary(monkeypatch):
    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: None)
    assert ct.start_studio_tunnel(8080) is None


def test_start_studio_tunnel_drops_url_that_is_not_publicly_reachable(monkeypatch):
    attempts = []

    class _Stub:
        def __init__(
            self,
            port,
            binary,
            protocol = None,
        ):
            self.url = None
            attempts.append(protocol)

        def start(self):
            self.url = "https://words.trycloudflare.com"

        def wait_for_ready(self, timeout):
            return self.url

        def stop(self):
            pass

    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: "/bin/cloudflared")
    monkeypatch.setattr(ct, "CloudflareTunnel", _Stub)
    monkeypatch.setattr(ct, "verify_public_url", lambda url, **kw: False)
    assert ct.start_studio_tunnel(8080) is None
    assert attempts == [None]
    assert ct._active_tunnel is None


def test_start_studio_tunnel_returns_url_once_probe_passes(monkeypatch):
    probed = []

    class _Stub:
        def __init__(
            self,
            port,
            binary,
            protocol = None,
        ):
            self.url = None
            self.protocol = protocol

        def start(self):
            self.url = "https://words.trycloudflare.com"

        def wait_for_ready(self, timeout):
            return self.url

        def stop(self):
            pass

    def _probe(url, **kw):
        probed.append(url)
        return True

    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: "/bin/cloudflared")
    monkeypatch.setattr(ct, "CloudflareTunnel", _Stub)
    monkeypatch.setattr(ct, "verify_public_url", _probe)
    try:
        assert ct.start_studio_tunnel(8080) == "https://words.trycloudflare.com"
        assert probed == ["https://words.trycloudflare.com"]
    finally:
        ct.stop_studio_tunnel()


def test_start_studio_tunnel_registers_before_wait(monkeypatch):
    # The tunnel must be visible to stop_studio_tunnel() during the readiness
    # wait, else a shutdown in that window orphans cloudflared.
    seen = {}

    class _Stub:
        def __init__(
            self,
            port,
            binary,
            protocol = None,
        ):
            self.url = None

        def start(self):
            pass

        def wait_for_ready(self, timeout):
            seen["active_during_wait"] = ct._active_tunnel is self
            self.url = "https://x.trycloudflare.com"
            return self.url

        def stop(self):
            seen["stopped"] = True

    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: "/bin/cloudflared")
    monkeypatch.setattr(ct, "CloudflareTunnel", _Stub)
    try:
        assert ct.start_studio_tunnel(8080) == "https://x.trycloudflare.com"
        assert seen["active_during_wait"] is True
    finally:
        ct.stop_studio_tunnel()


def test_start_studio_tunnel_clears_and_stops_on_no_url(monkeypatch):
    seen = {}

    class _Stub:
        def __init__(
            self,
            port,
            binary,
            protocol = None,
        ):
            self.url = None

        def start(self):
            pass

        def wait_for_ready(self, timeout):
            return None

        def stop(self):
            seen["stopped"] = True

    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: "/bin/cloudflared")
    monkeypatch.setattr(ct, "CloudflareTunnel", _Stub)
    assert ct.start_studio_tunnel(8080) is None
    assert seen.get("stopped") is True
    assert ct._active_tunnel is None


def test_start_studio_tunnel_returns_url(monkeypatch):
    class _StubTunnel:
        def __init__(
            self,
            port,
            binary,
            protocol = None,
        ):
            self.url = None

        def start(self):
            self.url = "https://stub-xyz.trycloudflare.com"

        def wait_for_ready(self, timeout):
            return self.url

        def stop(self):
            pass

    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: "/bin/cloudflared")
    monkeypatch.setattr(ct, "CloudflareTunnel", _StubTunnel)
    try:
        assert ct.start_studio_tunnel(8080) == "https://stub-xyz.trycloudflare.com"
    finally:
        ct.stop_studio_tunnel()


def test_start_studio_tunnel_falls_back_to_http2(monkeypatch):
    # First attempt mints a URL but never registers (quic blocked); the http2
    # retry registers and wins.
    attempts = []

    class _Stub:
        def __init__(
            self,
            port,
            binary,
            protocol = None,
        ):
            self.protocol = protocol
            self.url = None
            attempts.append(protocol)

        def start(self):
            self.url = "https://words.trycloudflare.com"  # URL always minted

        def wait_for_ready(self, timeout):
            return self.url if self.protocol == "http2" else None

        def stop(self):
            pass

    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: "/bin/cloudflared")
    monkeypatch.setattr(ct, "CloudflareTunnel", _Stub)
    try:
        assert ct.start_studio_tunnel(8080) == "https://words.trycloudflare.com"
        assert attempts == [None, "http2"]  # default first, then forced http2
    finally:
        ct.stop_studio_tunnel()


def test_start_studio_tunnel_no_retry_when_shutdown_between_attempts(monkeypatch):
    # A stop() landing in the gap AFTER the failed first attempt is cleaned up but
    # BEFORE the http2 retry registers must abort the loop -- not start a second
    # tunnel that nobody will ever stop (Codex review). Simulated by having the
    # first attempt's stop() (called during cleanup) trigger the shutdown.
    attempts = []

    class _Stub:
        def __init__(
            self,
            port,
            binary,
            protocol = None,
        ):
            self.url = None
            attempts.append(protocol)

        def start(self):
            self.url = "https://words.trycloudflare.com"  # URL minted, never ready

        def wait_for_ready(self, timeout):
            return None

        def stop(self):
            ct.stop_studio_tunnel()  # a concurrent shutdown lands in the gap

    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: "/bin/cloudflared")
    monkeypatch.setattr(ct, "CloudflareTunnel", _Stub)
    assert ct.start_studio_tunnel(8080) is None
    assert attempts == [None]  # http2 retry aborted after shutdown
    assert ct._active_tunnel is None


def test_start_studio_tunnel_no_http2_retry_when_no_url(monkeypatch):
    # No URL at all is an API/network failure; the http2 fallback would not help,
    # so it must be skipped (don't burn a second timeout window).
    attempts = []

    class _Stub:
        def __init__(
            self,
            port,
            binary,
            protocol = None,
        ):
            self.url = None
            attempts.append(protocol)

        def start(self):
            pass  # never mints a URL

        def wait_for_ready(self, timeout):
            return None

        def stop(self):
            pass

    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: "/bin/cloudflared")
    monkeypatch.setattr(ct, "CloudflareTunnel", _Stub)
    assert ct.start_studio_tunnel(8080) is None
    assert attempts == [None]


def test_start_studio_tunnel_both_protocols_fail_registration(monkeypatch):
    # Both quic and http2 mint a URL but neither registers -> both attempts are
    # exhausted and None is returned (no dead URL advertised).
    attempts = []

    class _Stub:
        def __init__(
            self,
            port,
            binary,
            protocol = None,
        ):
            self.url = None
            attempts.append(protocol)

        def start(self):
            self.url = "https://words.trycloudflare.com"  # URL minted, never ready

        def wait_for_ready(self, timeout):
            return None

        def stop(self):
            pass

    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: "/bin/cloudflared")
    monkeypatch.setattr(ct, "CloudflareTunnel", _Stub)
    assert ct.start_studio_tunnel(8080) is None
    assert attempts == [None, "http2"]
    assert ct._active_tunnel is None


def test_start_studio_tunnel_aborts_retry_on_concurrent_shutdown(monkeypatch):
    # If a concurrent stop_studio_tunnel() clears _active_tunnel while we wait,
    # the retry loop must NOT start a second (http2) tunnel: shutdown is already
    # done, so nothing would ever stop it and it would be orphaned.
    attempts = []

    class _Stub:
        def __init__(
            self,
            port,
            binary,
            protocol = None,
        ):
            self.url = None
            attempts.append(protocol)

        def start(self):
            self.url = "https://words.trycloudflare.com"  # URL minted (saw_url True)

        def wait_for_ready(self, timeout):
            # Simulate stop_studio_tunnel() landing during the wait.
            with ct._active_lock:
                ct._active_tunnel = None
            return None  # never registered

        def stop(self):
            pass

    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: "/bin/cloudflared")
    monkeypatch.setattr(ct, "CloudflareTunnel", _Stub)
    assert ct.start_studio_tunnel(8080) is None
    assert attempts == [None]  # no http2 retry -> no orphaned second tunnel
    assert ct._active_tunnel is None


# ── run.py source-level pins (AST / source, no heavy import) ─────────


def _func_param_defaults(source, func_name):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            args = node.args.args
            defaults = node.args.defaults
            offset = len(args) - len(defaults)
            out = {}
            for i, d in enumerate(defaults):
                if isinstance(d, ast.Constant):
                    out[args[offset + i].arg] = d.value
            return out
    return {}


def _argparse_default(source, option):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument" and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Constant) and a0.value == option:
                    for kw in node.keywords:
                        if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                            return kw.value.value
    return None


def test_run_server_cloudflare_default_off():
    defaults = _func_param_defaults(_RUN_PY.read_text(), "run_server")
    assert "cloudflare" in defaults
    assert defaults["cloudflare"] is None


def test_argparse_cloudflare_default_off():
    assert _argparse_default(_RUN_PY.read_text(), "--cloudflare") is None


def test_verify_global_reachability_marks_private_address_unreachable():
    src = _RUN_PY.read_text()
    tree = ast.parse(src)
    func_src = next(
        ast.get_source_segment(src, n)
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_verify_global_reachability"
    )
    captured = []
    ns = {
        "_public_reachable": None,
        "_stdout_color_ok": lambda: False,
        "_url_host": lambda host: host,
        "print": lambda *a, **k: captured.append(" ".join(str(x) for x in a)),
    }
    exec(compile(func_src, "<verify_global_reachability>", "exec"), ns)
    ns["_verify_global_reachability"]("192.168.1.10", 8888)

    assert ns["_public_reachable"] is False
    assert "private/LAN address" in "\n".join(captured)


def test_run_server_registers_tunnel_atexit_backstop():
    # An abnormal exit (exception after startup -> sys.exit) bypasses
    # _graceful_shutdown; an atexit backstop must still stop the tunnel.
    src = _RUN_PY.read_text()
    assert "atexit.register(stop_studio_tunnel)" in src


def _run_print_cloudflare_line(
    monkeypatch,
    *,
    cloudflare_url,
    public_reachable,
    cloudflare_requested = False,
    cloudflare_flag = True,
    secure = False,
    loopback_host = "127.0.0.1",
    color = False,
):
    """Exec _print_cloudflare_line without importing run.py's heavy deps."""
    src = _RUN_PY.read_text()
    tree = ast.parse(src)
    func_src = next(
        ast.get_source_segment(src, n)
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_print_cloudflare_line"
    )
    stub = types.ModuleType("startup_banner")
    stub.stdout_supports_color = lambda: color
    monkeypatch.setitem(sys.modules, "startup_banner", stub)
    captured: list[str] = []
    ns = {
        "_cloudflare_url": cloudflare_url,
        "_public_reachable": public_reachable,
        "_cloudflare_requested": cloudflare_requested,
        "_cloudflare_flag": cloudflare_flag,
        "print": lambda *a, **k: captured.append(" ".join(str(x) for x in a)),
    }
    exec(compile(func_src, "<print_cloudflare_line>", "exec"), ns)
    ns["_print_cloudflare_line"](secure = secure, loopback_host = loopback_host)
    return "\n".join(captured)


def test_cloudflare_line_reworded_when_public_unreachable(monkeypatch):
    out = _run_print_cloudflare_line(
        monkeypatch, cloudflare_url = "https://x.trycloudflare.com", public_reachable = False
    )
    assert "Use the secure link access via Cloudflare instead: https://x.trycloudflare.com" in out


def test_cloudflare_line_default_wording_when_reachable(monkeypatch):
    out = _run_print_cloudflare_line(
        monkeypatch, cloudflare_url = "https://x.trycloudflare.com", public_reachable = True
    )
    assert "Secure link access via Cloudflare: https://x.trycloudflare.com" in out
    assert "Use the secure link" not in out


def test_cloudflare_line_default_wording_when_unknown(monkeypatch):
    out = _run_print_cloudflare_line(
        monkeypatch, cloudflare_url = "https://x.trycloudflare.com", public_reachable = None
    )
    assert "Secure link access via Cloudflare: https://x.trycloudflare.com" in out
    assert "Use the secure link" not in out


def test_cloudflare_line_states_inactive_when_enabled_but_not_requested(monkeypatch):
    out = _run_print_cloudflare_line(monkeypatch, cloudflare_url = None, public_reachable = False)
    assert "Cloudflare tunnel: OFF for this mode" in out
    assert "local network only" in out


def test_cloudflare_line_warns_when_public_url_up(monkeypatch):
    out = _run_print_cloudflare_line(
        monkeypatch,
        cloudflare_url = "https://x.trycloudflare.com",
        public_reachable = True,
        cloudflare_requested = True,
    )
    assert "Secure link access via Cloudflare: https://x.trycloudflare.com" in out
    assert "Cloudflare tunnel: ON" in out
    assert "PUBLIC" in out
    assert "--no-cloudflare" in out
    assert "raw port is also publicly reachable" in out
    assert "local network only" not in out


def test_cloudflare_line_secure_mode_suppresses_public_warning(monkeypatch):
    out = _run_print_cloudflare_line(
        monkeypatch,
        cloudflare_url = "https://x.trycloudflare.com",
        public_reachable = True,
        cloudflare_requested = True,
        secure = True,
    )
    assert "Secure link access via Cloudflare: https://x.trycloudflare.com" in out
    assert "Cloudflare tunnel: ON" not in out


def test_cloudflare_line_states_disabled_when_off(monkeypatch):
    out = _run_print_cloudflare_line(
        monkeypatch,
        cloudflare_url = None,
        public_reachable = False,
        cloudflare_requested = False,
        cloudflare_flag = False,
    )
    assert "Cloudflare tunnel: OFF" in out
    assert "local network only" in out


def test_cloudflare_line_labels_unset_as_default(monkeypatch):
    # None = off by default (no flag) -> banner says "(default)", not "(--no-cloudflare)".
    out = _run_print_cloudflare_line(
        monkeypatch,
        cloudflare_url = None,
        public_reachable = False,
        cloudflare_requested = False,
        cloudflare_flag = None,
    )
    assert "Cloudflare tunnel: OFF (default)" in out
    assert "--no-cloudflare" not in out


def test_cloudflare_line_labels_explicit_no_cloudflare(monkeypatch):
    # False = explicit --no-cloudflare -> banner says "(--no-cloudflare)".
    out = _run_print_cloudflare_line(
        monkeypatch,
        cloudflare_url = None,
        public_reachable = False,
        cloudflare_requested = False,
        cloudflare_flag = False,
    )
    assert "Cloudflare tunnel: OFF (--no-cloudflare)" in out


def test_cloudflare_line_states_failed_when_requested_but_no_url(monkeypatch):
    out = _run_print_cloudflare_line(
        monkeypatch,
        cloudflare_url = None,
        public_reachable = False,
        cloudflare_requested = True,
        cloudflare_flag = True,
    )
    assert "requested but failed to start" in out
    assert "local network only" in out


def test_cloudflare_line_off_does_not_claim_local_only_when_unknown(monkeypatch):
    out = _run_print_cloudflare_line(
        monkeypatch,
        cloudflare_url = None,
        public_reachable = None,
        cloudflare_requested = False,
        cloudflare_flag = False,
    )
    assert "Cloudflare tunnel: OFF" in out
    assert "Raw port reachability was not verified" in out
    assert "local network only" not in out


def test_cloudflare_line_failed_does_not_claim_local_only_when_unknown(monkeypatch):
    out = _run_print_cloudflare_line(
        monkeypatch,
        cloudflare_url = None,
        public_reachable = None,
        cloudflare_requested = True,
        cloudflare_flag = True,
    )
    assert "requested but failed to start" in out
    assert "Raw port reachability was not verified" in out
    assert "local network only" not in out


@pytest.mark.parametrize(
    "cloudflare_requested,cloudflare_flag,expected",
    [
        (True, True, "requested but failed to start"),
        (False, True, "Cloudflare tunnel: OFF for this mode"),
        (False, False, "Cloudflare tunnel: OFF"),
    ],
)
def test_cloudflare_line_unknown_warns_with_loopback_host(
    monkeypatch, cloudflare_requested, cloudflare_flag, expected
):
    out = _run_print_cloudflare_line(
        monkeypatch,
        cloudflare_url = None,
        public_reachable = None,
        cloudflare_requested = cloudflare_requested,
        cloudflare_flag = cloudflare_flag,
        loopback_host = "::1",
        color = True,
    )
    assert expected in out
    assert "bind ::1" in out
    assert "bind 127.0.0.1" not in out
    assert "\033[38;5;215;1m" in out


def test_cloudflare_line_off_does_not_claim_local_only_when_publicly_reachable(monkeypatch):
    out = _run_print_cloudflare_line(
        monkeypatch,
        cloudflare_url = None,
        public_reachable = True,
        cloudflare_requested = False,
        cloudflare_flag = False,
    )
    assert "Cloudflare tunnel: OFF" in out
    assert "reachable from the public internet" in out
    assert "local network only" not in out


def test_cloudflare_line_failed_does_not_claim_local_only_when_publicly_reachable(monkeypatch):
    out = _run_print_cloudflare_line(
        monkeypatch,
        cloudflare_url = None,
        public_reachable = True,
        cloudflare_requested = True,
        cloudflare_flag = True,
    )
    assert "requested but failed to start" in out
    assert "reachable from the public internet" in out
    assert "local network only" not in out
