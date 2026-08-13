# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the Cloudflare temporary-tunnel helper and run.py wiring.

cloudflare_tunnel.py is stdlib-only (storage_roots is imported lazily), so it
loads via spec_from_file_location without the studio venv. run.py defaults are
checked by AST so we never import its heavy deps (uvicorn/structlog).
"""

import ast
import importlib.util
import io
import os
import subprocess
import sys
import tarfile
import threading
import time
import types
from pathlib import Path
from typing import Optional

import pytest

from utils import process_lifetime

_BACKEND = Path(__file__).resolve().parent.parent
_CT_PY = _BACKEND / "cloudflare_tunnel.py"
_RUN_PY = _BACKEND / "run.py"


def _load_ct():
    spec = importlib.util.spec_from_file_location("cloudflare_tunnel", _CT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ct = _load_ct()


def test_spawn_login_disables_cloudflared_browser_launcher(monkeypatch):
    captured = {}
    monkeypatch.setattr(ct, "_spawn_child", lambda spawn: spawn())
    monkeypatch.setattr(
        ct.subprocess,
        "Popen",
        lambda argv, **kwargs: captured.update(argv = argv, **kwargs) or object(),
    )
    ct._spawn_login("/cloudflared", "operation-token")
    assert captured["env"]["PATH"] == ""
    assert captured["env"]["NoDefaultCurrentDirectoryInExePath"] == "1"
    assert captured["env"][ct._TOKEN_VAR] == "operation-token"


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
    # as the tunnel URL (it returns a 404 and is not a temporary tunnel).
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
        self.pid = 424243  # every real Popen has one; the lifetime record reads it

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


def test_runtime_callback_covers_process_start_through_stop(monkeypatch):
    events = []
    starts_admitted = []
    ct.set_studio_tunnel_runtime_callback(events.append)

    class _ObservedPopen(_FakePopen):
        stdout = None
        block_termination = True

        def terminate(self):
            assert events[-1] is True
            if self.block_termination:
                ct.start_studio_tunnel(8081, managed_by = "settings")
                raise OSError("termination unavailable")
            super().terminate()

    fake = _ObservedPopen()

    def fake_popen(*_args, **_kwargs):
        assert events[-1] is True
        return fake

    class _NoReaderThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(ct.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ct.threading, "Thread", _NoReaderThread)
    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: starts_admitted.append(True))
    try:
        tunnel = ct.CloudflareTunnel(8080, "/bin/cloudflared")
        tunnel.start()
        assert events[-1] is True
        ct._active_tunnel = tunnel
        ct._tunnel_state = "online"
        ct._tunnel_owner = "settings"
        ct._tunnel_port = 8080
        ct._tunnel_state = "stopping"
        ct._active_tunnel_exited(tunnel)
        assert ct.get_studio_tunnel_status()["state"] == "stopping"
        ct._tunnel_state = "online"
        ct._active_tunnel_exited(tunnel)
        assert events[-1] is True
        assert starts_admitted == []
        assert ct._active_tunnel is tunnel
        assert ct.get_studio_tunnel_status()["state"] == "error"
        assert ct.get_studio_tunnel_status()["stop_pending"] is True
        fake.block_termination = False
        ct.stop_studio_tunnel()
        assert events[-1] is False
        assert ct.get_studio_tunnel_status()["state"] == "off"
        ct._retain_studio_tunnel_for_stop(tunnel)
        assert ct._tunnels_pending_stop_snapshot() == ()
    finally:
        ct.set_studio_tunnel_runtime_callback(None)


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
    exited = []
    t.set_on_exit(exited.append)
    t._reader(
        _fake_proc(
            "INF Requesting new quick Tunnel on trycloudflare.com...\n"
            "INF |  https://words-here-abc.trycloudflare.com  |\n"
            "INF Registered tunnel connection connIndex=0 protocol=http2\n"
        )
    )
    assert t.url == "https://words-here-abc.trycloudflare.com"
    assert t.ready is True
    assert t.wait_for_ready(0) is None
    assert t.error == "cloudflared exited"
    assert exited == [t]


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
def _stub_remote_lookups(monkeypatch, request):
    if request.node.name.startswith("test_verify_public_url"):
        monkeypatch.setattr(ct, "_wait_for_dns", lambda *a, **kw: None)
        monkeypatch.setattr(ct, "_edge_addresses", list)


def test_wait_for_dns_polls_until_answer(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req.full_url)
        if len(calls) < 3:
            return _FakeResponse(b'{"Status":3}')
        return _FakeResponse(b'{"Status":0,"Answer":[{"data":"104.16.0.1"}]}')

    _patch_urlopen(monkeypatch, handler)
    monkeypatch.setattr(ct.time, "sleep", lambda _s: None)
    assert ct._wait_for_dns("words.trycloudflare.com", ct.time.monotonic() + 5) is True
    assert len(calls) == 3
    assert "name=words.trycloudflare.com" in calls[0]
    # The tunnel provider already knows the hostname it just issued; no one else does.
    assert all("cloudflare-dns.com" in call for call in calls)


def test_wait_for_dns_gives_up_at_deadline(monkeypatch):
    _patch_urlopen(monkeypatch, lambda req: _FakeResponse(b'{"Status":3}'))
    monkeypatch.setattr(ct.time, "sleep", lambda _s: None)
    assert ct._wait_for_dns("words.trycloudflare.com", ct.time.monotonic() + 0.05) is False


def test_wait_for_dns_retries_transient_doh_error(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req.full_url)
        if len(calls) < 3:
            raise OSError("transient")
        return _FakeResponse(b'{"Status":0,"Answer":[{"data":"104.16.0.1"}]}')

    _patch_urlopen(monkeypatch, handler)
    monkeypatch.setattr(ct.time, "sleep", lambda _s: None)
    ct._wait_for_dns("words.trycloudflare.com", ct.time.monotonic() + 5)
    assert len(calls) == 3


def test_wait_for_dns_bails_on_persistent_doh_errors(monkeypatch):
    calls = []

    def handler(req):
        calls.append(req.full_url)
        raise OSError("blocked")

    _patch_urlopen(monkeypatch, handler)
    monkeypatch.setattr(ct.time, "sleep", lambda _s: None)
    assert ct._wait_for_dns("words.trycloudflare.com", ct.time.monotonic() + 5) is None
    assert len(calls) == ct._DNS_MAX_DOH_ERRORS
    # Blocking one of these services is how such a network is configured, so the
    # attempts keep alternating rather than settling on whichever failed last.
    assert [call.split("?")[0] for call in calls] == [
        ct._DOH_URLS[index % len(ct._DOH_URLS)].split("?")[0]
        for index in range(ct._DNS_MAX_DOH_ERRORS)
    ]


def test_wait_for_dns_takes_the_answer_of_whichever_service_is_reachable(monkeypatch):
    asked = []

    def handler(req):
        asked.append(req.full_url)
        if req.full_url.startswith(ct._DOH_URLS[0].split("?")[0]):
            raise OSError("blocked")
        return _FakeResponse(b'{"Status":0,"Answer":[{"data":"203.0.113.7"}]}')

    _patch_urlopen(monkeypatch, handler)
    monkeypatch.setattr(ct.time, "sleep", lambda _s: None)
    assert ct._wait_for_dns("studio.example.com", ct.time.monotonic() + 5) is True
    assert all("studio.example.com" in call for call in asked)


def test_wait_for_dns_delays_first_query(monkeypatch):
    order = []

    def handler(req):
        order.append("query")
        return _FakeResponse(b'{"Status":0,"Answer":[{"data":"104.16.0.1"}]}')

    _patch_urlopen(monkeypatch, handler)
    monkeypatch.setattr(ct.time, "sleep", lambda s: order.append(("sleep", s)))
    ct._wait_for_dns("words.trycloudflare.com", ct.time.monotonic() + 30)
    # a token hold-off would not outlast the propagation that makes the first query miss
    assert ct._DNS_INITIAL_GRACE >= 1.0
    assert order[0] == ("sleep", ct._DNS_INITIAL_GRACE)
    assert order[1] == "query"


def test_wait_for_dns_is_capped_below_the_probe_deadline(monkeypatch):
    clock = [0.0]
    calls = []

    def handler(req):
        calls.append(req.full_url)
        return _FakeResponse(b'{"Status":3}')

    _patch_urlopen(monkeypatch, handler)
    monkeypatch.setattr(ct.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ct.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    ct._wait_for_dns("words.trycloudflare.com", 300.0)
    assert calls
    assert clock[0] <= ct._DNS_WAIT_MAX + ct._DNS_POLL_DELAY
    # The probe shares one deadline with the wait and needs most of it.
    assert clock[0] < ct._PUBLIC_PROBE_TIMEOUT / 2


def test_verify_public_url_accepts_studio_marker(monkeypatch):
    seen = {}

    def handler(req):
        seen["url"] = req.full_url
        return _FakeResponse(b'{"status":"healthy","service":"Unsloth UI Backend"}')

    _patch_urlopen(monkeypatch, handler)
    assert ct.verify_public_url("https://words.trycloudflare.com") is True
    assert seen["url"] == "https://words.trycloudflare.com/api/health"


def test_verify_public_url_waits_for_dns_before_probing_the_hostname(monkeypatch):
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


def _fake_edge(
    monkeypatch,
    payload: bytes,
    connect_error: Optional[str] = None,
) -> dict:
    """Stand in for the TLS hop so the probe's own parsing is under test."""
    import socket as socket_module
    import ssl as ssl_module

    seen: dict = {}

    class _Closeable:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

    class _Tls(_Closeable):
        def sendall(self, data):
            seen["request"] = data

        def makefile(self, *_a, **_kw):
            return io.BytesIO(payload)

    class _Context:
        # A default context verifies the chain and matches it against the SNI
        # name; the probe dials a bare address, so that match is the only thing
        # binding the answer to the tunnel.
        check_hostname = True
        verify_mode = ssl_module.CERT_REQUIRED

        def wrap_socket(
            self,
            _raw,
            server_hostname = None,
        ):
            seen["sni"] = server_hostname
            seen["verified"] = self.check_hostname and self.verify_mode == ssl_module.CERT_REQUIRED
            return _Tls()

    def connect(address, timeout = None):
        seen["timeout"] = timeout
        if connect_error is not None:
            raise OSError(connect_error)
        seen["address"] = address
        return _Closeable()

    monkeypatch.setattr(socket_module, "create_connection", connect)
    monkeypatch.setattr(ssl_module, "create_default_context", _Context)
    return seen


def _edge_response(status: bytes, body: bytes) -> bytes:
    return b"HTTP/1.1 %s\r\nContent-Length: %d\r\n\r\n%s" % (status, len(body), body)


def test_edge_probe_selects_the_tunnel_by_sni(monkeypatch):
    seen = _fake_edge(monkeypatch, _edge_response(b"200 OK", b'{"service":"Unsloth UI Backend"}'))
    assert ct._probe_edge("104.16.0.1", "words.trycloudflare.com") is True
    assert seen["address"] == ("104.16.0.1", 443)
    # Cloudflare picks the tunnel from SNI and the Host header, not the address.
    assert seen["sni"] == "words.trycloudflare.com"
    assert seen["verified"] is True
    assert seen["timeout"] == ct._PUBLIC_PROBE_ATTEMPT_TIMEOUT
    assert seen["request"].startswith(f"GET {ct._PUBLIC_PROBE_PATH} HTTP/1.1\r\n".encode())
    assert b"Host: words.trycloudflare.com\r\n" in seen["request"]


def test_edge_probe_rejects_the_cloudflare_error_page(monkeypatch):
    _fake_edge(monkeypatch, _edge_response(b"530 ", b"<html>error 1033</html>"))
    assert ct._probe_edge("104.16.0.1", "words.trycloudflare.com") is False


def test_edge_probe_rejects_a_foreign_responder(monkeypatch):
    # Well-formed JSON from something that is not this backend, e.g. a proxy.
    _fake_edge(monkeypatch, _edge_response(b"200 OK", b'{"service":"something else"}'))
    assert ct._probe_edge("104.16.0.1", "words.trycloudflare.com") is False


def test_edge_addresses_keep_one_entry_per_frontend(monkeypatch):
    import socket as socket_module

    # macOS reports the A records mapped into IPv6; both forms are one frontend.
    resolved = [
        (socket_module.AF_INET, 1, 6, "", ("104.16.230.132", 443)),
        (socket_module.AF_INET6, 1, 6, "", ("::ffff:104.16.230.132", 443, 0, 0)),
        (socket_module.AF_INET6, 1, 6, "", ("2606:4700::6810:e684", 443, 0, 0)),
    ]
    monkeypatch.setattr(socket_module, "getaddrinfo", lambda *_a, **_kw: resolved)
    assert ct._edge_addresses() == ["104.16.230.132", "2606:4700::6810:e684"]


def test_edge_probe_reports_an_unreachable_edge_apart_from_a_wrong_answer(monkeypatch):
    _fake_edge(monkeypatch, b"", connect_error = "blocked")
    assert ct._probe_edge("104.16.0.1", "words.trycloudflare.com") is None


def test_edge_verification_skips_the_hostname_entirely(monkeypatch):
    seen = {}

    def probe(
        address,
        host,
        _timeout = None,
    ):
        seen["probe"] = (address, host)
        return True

    def resolve(*_a, **_kw):
        seen["dns"] = True

    monkeypatch.setattr(ct, "_edge_addresses", lambda: ["104.16.0.1"])
    monkeypatch.setattr(ct, "_probe_edge", probe)
    monkeypatch.setattr(ct, "_wait_for_dns", resolve)
    _patch_urlopen(monkeypatch, lambda req: pytest.fail("probed the hostname"))
    assert ct.verify_public_url("https://words.trycloudflare.com") is True
    assert seen["probe"] == ("104.16.0.1", "words.trycloudflare.com")
    assert "dns" not in seen


def test_edge_verification_polls_until_the_tunnel_answers(monkeypatch):
    # Error 1033 and a proxy's own page answer, so a reply without the marker is
    # not an unreachable edge and must not count towards giving up.
    answers = [False, None, False, None, True]
    monkeypatch.setattr(ct, "_edge_addresses", lambda: ["104.16.0.1"])
    monkeypatch.setattr(ct, "_probe_edge", lambda *_a: answers.pop(0))
    monkeypatch.setattr(ct.time, "sleep", lambda _s: None)
    assert ct._verify_through_edge("words.trycloudflare.com", ct.time.monotonic() + 5) is True
    assert not answers


def test_edge_verification_stops_at_the_deadline_mid_pass(monkeypatch):
    clock = [0.0]
    timeouts = []

    def probe(_address, _host, timeout):
        timeouts.append(timeout)
        clock[0] += 6.0
        return False

    monkeypatch.setattr(ct, "_edge_addresses", lambda: ["104.16.0.1", "104.16.0.2"])
    monkeypatch.setattr(ct, "_probe_edge", probe)
    monkeypatch.setattr(ct.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ct.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    assert ct._verify_through_edge("words.trycloudflare.com", 3.0) is False
    # One attempt, given only the time the deadline leaves: the second address is
    # already past it, and a whole pass must not outlive the caller's budget.
    assert timeouts == [3.0]


def test_edge_verification_leaves_the_fallback_room_in_the_deadline(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(ct, "_edge_addresses", lambda: ["104.16.0.1"])
    monkeypatch.setattr(ct, "_probe_edge", lambda *_a: False)
    monkeypatch.setattr(ct.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ct.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    assert ct._verify_through_edge("words.trycloudflare.com", 300.0) is False
    assert clock[0] <= ct._EDGE_WAIT_MAX + ct._EDGE_PROBE_RETRY_DELAY
    # What the cap is for: the hostname fallback still needs its DNS wait and
    # several attempts of its own out of the one shared deadline.
    left = ct._PUBLIC_PROBE_TIMEOUT - ct._EDGE_WAIT_MAX - ct._DNS_WAIT_MAX
    assert left >= 5 * ct._PUBLIC_PROBE_RETRY_DELAY


def test_blocked_edge_falls_back_to_the_hostname(monkeypatch):
    order = []

    def probe(*_a):
        order.append("edge")
        return None

    def handler(_req):
        order.append("hostname")
        return _FakeResponse(b'{"service":"Unsloth UI Backend"}')

    monkeypatch.setattr(ct, "_edge_addresses", lambda: ["104.16.0.1"])
    monkeypatch.setattr(ct, "_probe_edge", probe)
    monkeypatch.setattr(ct, "_wait_for_dns", lambda *_a: order.append("dns"))
    _patch_urlopen(monkeypatch, handler)
    monkeypatch.setattr(ct.time, "sleep", lambda _s: None)
    assert ct.verify_public_url("https://words.trycloudflare.com") is True
    assert order == ["edge"] * ct._EDGE_MAX_UNREACHABLE + ["dns", "hostname"]


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


def test_custom_start_explains_when_the_registered_hostname_is_unreachable(monkeypatch):
    class _Reservation:
        token = "token"

        def release(self):
            pass

    class _Stub:
        def __init__(self, _port, _binary, **kwargs):
            self.url = kwargs["url"]

        def start(self):
            pass

        def wait_for_ready(self, _timeout):
            return self.url

        def stop(self):
            return True

    identity = {
        "hostname": "studio.example.com",
        "tunnel_name": "unsloth-AB12CD",
        "tunnel_id": _TUNNEL_ID,
        "credentials": "/tmp/credentials.json",
    }
    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: "/bin/cloudflared")
    monkeypatch.setattr(ct, "read_identity", lambda: identity)
    monkeypatch.setattr(ct, "identity_is_runnable", lambda _identity: True)
    monkeypatch.setattr(ct, "reserve_connector", _Reservation)
    monkeypatch.setattr(ct, "write_custom_ingress", lambda *_a, **_kw: Path("config.yml"))
    monkeypatch.setattr(ct, "custom_tunnel_args", lambda *_a: [])
    monkeypatch.setattr(ct, "CloudflareTunnel", _Stub)
    monkeypatch.setattr(ct, "verify_public_url", lambda *_a, **_kw: False)

    assert ct.start_studio_tunnel(8080, managed_by = "settings", kind = "custom") is None
    assert ct.get_studio_tunnel_status()["error"] == "custom_hostname_unreachable"


def test_starting_a_tunnel_starts_watching_its_hostname(monkeypatch):
    watched = []

    class _Stub:
        def __init__(
            self,
            port,
            binary,
            protocol = None,
        ):
            self.url = None

        def start(self):
            self.url = "https://app.example.com"

        def wait_for_ready(self, timeout):
            return self.url

        def stop(self):
            pass

    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: "/bin/cloudflared")
    monkeypatch.setattr(ct, "CloudflareTunnel", _Stub)
    monkeypatch.setattr(ct, "verify_public_url", lambda url, **kw: True)
    monkeypatch.setattr(
        ct, "_watch_hostname_resolution", lambda url, generation: watched.append((url, generation))
    )
    try:
        assert ct.start_studio_tunnel(8080) == "https://app.example.com"
        for _ in range(200):
            if watched:
                break
            time.sleep(0.01)
        assert [u for u, _g in watched] == ["https://app.example.com"]
        assert watched[0][1] == ct._tunnel_generation
    finally:
        ct.stop_studio_tunnel()


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


def test_tunnel_status_tracks_owner_and_post_ready_exit(monkeypatch):
    instances = []

    class _Stub:
        late_exit = False

        def __init__(self, *_args, **_kwargs):
            self.url = "https://words.trycloudflare.com"
            self.error = None
            instances.append(self)

        start = stop = lambda self: None
        wait_for_ready = lambda self, timeout: self.url

        def set_on_exit(self, callback):
            self.on_exit = callback
            if self.late_exit:
                self.error = "cloudflared exited"

        def _publish_if_running(self, callback):
            if self.late_exit:
                return False
            callback()
            return True

    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: "/bin/cloudflared")
    monkeypatch.setattr(ct, "CloudflareTunnel", _Stub)
    monkeypatch.setattr(ct, "verify_public_url", lambda url, **kw: True)
    published = []
    ct.set_studio_tunnel_url_callback(published.append)
    assert ct.start_studio_tunnel(8087, managed_by = "settings") == instances[0].url
    status = ct.get_studio_tunnel_status()
    assert status["state"] == "online"
    assert status["managed_by"] == "settings"
    assert status["url"] == instances[0].url
    assert status["port"] == 8087
    instances[0].error = "cloudflared exited"
    instances[0].on_exit(instances[0])
    status = ct.get_studio_tunnel_status()
    assert status["state"] == "error"
    assert status["url"] is None
    assert status["error"] == "cloudflared exited"
    assert published[-1] is None
    ct.stop_studio_tunnel()
    _Stub.late_exit = True
    published.clear()
    assert ct.start_studio_tunnel(8087, managed_by = "settings") is None
    assert instances[-1].url not in published
    assert ct.get_studio_tunnel_status()["state"] == "error"
    ct.set_studio_tunnel_url_callback(None)


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
    defaults = _func_param_defaults(_RUN_PY.read_text(encoding = "utf-8"), "run_server")
    assert "cloudflare" in defaults
    assert defaults["cloudflare"] is None


def test_argparse_cloudflare_default_off():
    assert _argparse_default(_RUN_PY.read_text(encoding = "utf-8"), "--cloudflare") is None


def test_verify_global_reachability_marks_private_address_unreachable():
    src = _RUN_PY.read_text(encoding = "utf-8")
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
    src = _RUN_PY.read_text(encoding = "utf-8")
    assert "atexit.register(close_studio_tunnel_lifecycle)" in src


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
    src = _RUN_PY.read_text(encoding = "utf-8")
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


class _WatchStopped(Exception):
    pass


def _one_watch_pass(monkeypatch, url, generation):
    monkeypatch.setattr(ct.time, "sleep", lambda _s: (_ for _ in ()).throw(_WatchStopped()))
    try:
        ct._watch_hostname_resolution(url, generation)
    except _WatchStopped:
        pass


@pytest.mark.parametrize(
    "kind,dns,usable",
    [
        ("custom", "unknown", False),
        ("custom", "pending", False),
        ("custom", "resolved", True),
        # A temporary hostname is Cloudflare's own and resolves before the URL
        # is ever reported, so there is nothing to wait for.
        ("temporary", "unknown", True),
    ],
)
def test_the_status_says_whether_the_url_can_be_offered_yet(monkeypatch, kind, dns, usable):
    # Everyone offering the link was deciding this for itself; the tunnel is the
    # one place that knows.
    monkeypatch.setattr(ct, "_tunnel_kind", kind)
    monkeypatch.setattr(ct, "_tunnel_generation", 21)
    monkeypatch.setattr(ct, "_tunnel_dns", (21, dns))
    status = ct.get_studio_tunnel_status()
    assert status["dns"] == dns
    assert status["url_usable"] is usable


def test_a_slower_watcher_does_not_erase_a_newer_tunnels_answer(monkeypatch):
    monkeypatch.setattr(ct, "_tunnel_dns", (0, "unknown"))
    monkeypatch.setattr(ct, "_tunnel_generation", 4)
    monkeypatch.setattr(ct, "_tunnel_url", "https://new.example.com")
    monkeypatch.setattr(ct, "_wait_for_dns", lambda *a, **k: True)
    ct._watch_hostname_resolution("https://new.example.com", 4)
    monkeypatch.setattr(ct, "_wait_for_dns", lambda *a, **k: False)
    _one_watch_pass(monkeypatch, "https://old.example.com", 3)
    assert ct.get_studio_tunnel_status()["dns"] == "resolved"


def test_an_expired_deadline_issues_no_request(monkeypatch):
    calls = []

    def blocked(req):
        calls.append(1)
        raise OSError("blocked")

    _patch_urlopen(monkeypatch, blocked)
    monkeypatch.setattr(ct.time, "sleep", lambda _s: None)
    assert ct._wait_for_dns("words.trycloudflare.com", ct.time.monotonic() - 1) is None
    assert calls == []


def test_the_readiness_probe_ignores_an_ambient_proxy(monkeypatch):
    import urllib.request

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    openers = []
    real_build = urllib.request.build_opener

    def build_opener(*handlers):
        opener = real_build(*handlers)
        openers.append(opener)
        opener.open = lambda *a, **k: (_ for _ in ()).throw(OSError("no connector here"))
        return opener

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    assert ct._connector_reports_ready("127.0.0.1:20241", 1.0) is False
    assert openers
    proxies = [
        h.proxies for o in openers for h in o.handlers if isinstance(h, urllib.request.ProxyHandler)
    ]
    assert not any(proxies)


def test_the_readiness_probe_answers_from_the_status_cloudflared_sends():
    import json
    from http.server import BaseHTTPRequestHandler, HTTPServer

    connections = [0]

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_GET(self):  # noqa: N802
            # cloudflared's own /ready: 200 once a connection is registered and
            # 503 before that. The last case is the same connected tunnel behind
            # a build that stopped reporting the count, which still serves.
            status = 200 if connections[0] else 503
            body = (
                "OK"
                if connections[0] == "counted out"
                else json.dumps({"status": status, "readyConnections": connections[0]})
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    address = f"127.0.0.1:{server.server_port}"
    try:
        assert ct._connector_reports_ready(address, 5.0) is False
        connections[0] = 1
        assert ct._connector_reports_ready(address, 5.0) is True
        connections[0] = "counted out"
        assert ct._connector_reports_ready(address, 5.0) is True
    finally:
        server.shutdown()
        server.server_close()


def test_the_watch_deadline_does_not_drift_outward(monkeypatch):
    monkeypatch.setattr(ct, "_tunnel_generation", 12)
    monkeypatch.setattr(ct, "_tunnel_url", "https://h.example.com")
    monkeypatch.setattr(ct, "_tunnel_dns", (0, "unknown"))
    monkeypatch.setattr(ct, "_DNS_WATCH_TOTAL", 1.0)
    given = []
    clock = [0.0]

    def monotonic():
        clock[0] += 0.1
        return clock[0]

    def wait(host, deadline):
        given.append(deadline)
        return False

    monkeypatch.setattr(ct, "_wait_for_dns", wait)
    monkeypatch.setattr(ct.time, "monotonic", monotonic)
    _one_watch_pass(monkeypatch, "https://h.example.com", 12)
    assert given and all(d <= 0.1 + ct._DNS_WATCH_TOTAL for d in given)


def test_a_watch_that_outran_its_total_publishes_nothing(monkeypatch):
    monkeypatch.setattr(ct, "_tunnel_generation", 14)
    monkeypatch.setattr(ct, "_tunnel_url", "https://h.example.com")
    monkeypatch.setattr(ct, "_tunnel_dns", (14, "unknown"))
    monkeypatch.setattr(ct, "_DNS_WATCH_TOTAL", 1.0)
    clock = [0.0]
    monkeypatch.setattr(ct.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(ct.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))

    def wait(host, deadline):
        clock[0] += ct._DNS_WATCH_TOTAL + 1.0
        return False

    monkeypatch.setattr(ct, "_wait_for_dns", wait)
    ct._watch_hostname_resolution("https://h.example.com", 14)
    assert ct._tunnel_dns == (14, "unknown")


_TUNNEL_ID = "11111111-2222-3333-4444-5555aabbccdd"
_HOST = "studio.example.com"
_LOGIN_URL, _PROMPT = "https://dash.fed.cloudflare.com/argotunnel?aud=x", "Please open this URL:"
_API_REFUSAL = "code: 1000, reason: Invalid access token"
_WRAP = (
    "failed to provision routing, please create it manually via Cloudflare dashboard or UI; "
    "most likely you already have a conflicting record there. You can also rerun this command "
    "with --overwrite-dns to overwrite any existing DNS records for this hostname.: "
)
_COLLIDED = "failed to create tunnel: tunnel with name already exists"
_OFFLINE = "INF Request failed\ndial tcp: i/o timeout"
_ABSENT = "there should only be 1 non-deleted Tunnel named unsloth-AB12CD"


class FakeLogin:
    def __init__(
        self,
        cert,
        *,
        rc = 0,
        alive = 0,
        extra = "",
    ):
        self.pid, self.records = 999_000, []
        self.stdout = iter([_PROMPT + "\n", _LOGIN_URL + "\n", extra])
        self.returncode, self._cert, self._rc, self._alive = None, cert, rc, alive

    def _write_cert(self, text):
        if self._cert is not None:
            self._cert.parent.mkdir(parents = True, exist_ok = True)
            self._cert.write_text(text, encoding = "utf-8")

    def poll(self):
        self.records.append(ct._read(ct._RECORD))
        if self.returncode is not None:
            return self.returncode
        self._write_cert("" if self._alive else "STUDIO-CERT")
        if self._alive > 0:
            self._alive -= 1
            return None
        self.returncode = self._rc
        return self.returncode

    def terminate(self):
        self._write_cert("STUDIO-CERT")
        self.returncode = -15

    def wait(self, timeout = None):
        return self.returncode


class FakeCloudflared:
    def __init__(self, cert_dir):
        self.cert_dir, self.calls, self.deleted = cert_dir, [], []
        self.cert_at_delete = []
        self.create_outcomes = []
        self.route_outcome = None
        self.delete_outcome = None
        self.login = lambda: FakeLogin(cert_dir / "cert.pem", alive = 1)
        self.child = None

    def spawn(self, binary, token):
        self.spawned_with = token
        self.child = self.login()
        return self.child

    def __call__(self, binary, *args):
        self.calls.append(args)
        outcome = (1, "unexpected command")
        if args[0] == "create":
            outcome = self.create_outcomes.pop(0) if self.create_outcomes else self._create(args[1])
        elif args[0] == "route" and args[1] == "dns" and ct._NAME_RE.match(args[2]):
            added = f"Added CNAME {args[-1]} which will route to this tunnel"
            outcome = self.route_outcome or (0, added)
        elif args[0] == "delete":
            self.deleted.append(args[1])
            self.cert_at_delete.append((self.cert_dir / "cert.pem").exists())
            outcome = self.delete_outcome or (0, "")
        out, err = ("", outcome[1]) if outcome[0] else (outcome[1], "")
        return subprocess.CompletedProcess(args, outcome[0], out, err)

    def _create(self, name):
        assert name in (ct._read(ct._RECORD) or {}).get("tunnel_names", [])
        credentials = self.cert_dir / "creds" / f"{_TUNNEL_ID}.json"
        credentials.parent.mkdir(parents = True, exist_ok = True)
        credentials.write_text("{}", encoding = "utf-8")
        return 0, (
            f"Created tunnel {name} with id {_TUNNEL_ID.upper()}\n"
            f"Tunnel credentials written to {credentials}."
        )


@pytest.fixture
def cf(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    cert_dir = tmp_path / "cloudflared"
    monkeypatch.setattr(ct, "origin_cert_path", lambda: cert_dir / "cert.pem")
    monkeypatch.setattr(ct, "_claim_path", lambda: tmp_path / "claim" / "claim.lock")
    fake = FakeCloudflared(cert_dir)
    monkeypatch.setattr(ct, "_cli", fake)
    monkeypatch.setattr(ct, "_spawn_login", fake.spawn)
    return fake


def _provision(hostname = _HOST, **kwargs):
    return ct.provision_custom_tunnel(hostname, binary = "cloudflared", **kwargs)


def _settle(binary = "cloudflared"):
    with ct.certificate_state_claim("cleanup"):
        return ct._settle(binary)


def _cert(cf):
    return cf.cert_dir / "cert.pem"


def _created(cf):
    return [call[1] for call in cf.calls if call[0] == "create"]


def test_a_successful_run_records_the_identity_and_removes_the_certificate(cf):
    seen = []
    identity = _provision("https://Studio.Example.COM/", on_login_url = seen.append)
    assert seen == [_LOGIN_URL]
    assert identity["hostname"] == _HOST
    assert ct._NAME_RE.match(identity["tunnel_name"])
    assert identity["tunnel_id"] == _TUNNEL_ID
    assert not _cert(cf).exists()
    assert ct._read(ct._RECORD) is None
    assert ct._string_list(ct._ORPHANS) == []
    credentials = ct.state_root() / f"{_TUNNEL_ID}.json"
    assert identity["credentials"] == str(credentials)
    assert credentials.stat().st_mode & 0o777 == 0o600
    assert not any({"--overwrite-dns", "-f"} & set(call) for call in cf.calls)
    assert 999_000 not in process_lifetime._tracked_pids


def test_a_tunnel_whose_credentials_never_arrived_is_refused_before_the_route(cf):
    # Routing and writing an identity here would leave setup blocked by that
    # identity while nothing could start the tunnel it names.
    missing = cf.cert_dir / "creds" / f"{_TUNNEL_ID}.json"
    cf.create_outcomes = [
        (0, f"Created tunnel x with id {_TUNNEL_ID}\nTunnel credentials written to {missing}.")
    ]
    with pytest.raises(ct.ProvisioningError) as excinfo:
        _provision()
    assert excinfo.value.code == "credentials_missing"
    assert ct.read_identity() is None
    assert not any(call[0] == "route" for call in cf.calls)
    assert ct._string_list(ct._ORPHANS) == []


def test_a_certificate_that_cannot_be_removed_keeps_the_record_that_owns_it(cf, monkeypatch):
    # The digest is the only proof Studio wrote this account-wide file, so a
    # failed deletion has to keep the record rather than strand the file.
    real_unlink = ct._unlink
    monkeypatch.setattr(
        ct,
        "_unlink",
        lambda path, **kw: None if path == _cert(cf) else real_unlink(path, **kw),
    )

    identity = _provision()
    assert identity["hostname"] == _HOST
    assert _cert(cf).exists()
    record = ct._read(ct._RECORD)
    assert record["cert_digest"]

    # Once the file can be removed the retained record settles it at next launch.
    monkeypatch.setattr(ct, "_unlink", real_unlink)
    _settle()
    assert not _cert(cf).exists()
    assert ct._read(ct._RECORD) is None


def test_credentials_a_failed_setup_cannot_remove_keep_the_record_that_names_them(cf, monkeypatch):
    # The credentials authorize serving a tunnel that may have outlived the
    # failure, and only this record carries the path to them.
    credentials = ct.state_root() / f"{_TUNNEL_ID}.json"
    real_unlink = ct._unlink
    monkeypatch.setattr(
        ct,
        "_unlink",
        lambda path, **kw: None if path == credentials else real_unlink(path, **kw),
    )
    cf.route_outcome = (1, "failed to find zone")
    with pytest.raises(ct.ProvisioningError):
        _provision()
    assert credentials.exists()
    assert ct._read(ct._RECORD)["credentials"] == str(credentials)

    monkeypatch.setattr(ct, "_unlink", real_unlink)
    _settle()
    assert not credentials.exists()
    assert ct._read(ct._RECORD) is None


def test_teardown_carries_the_digest_of_a_certificate_setup_could_not_remove(cf, monkeypatch):
    # Teardown replaces the retained record, so dropping the digest here would
    # leave the certificate on disk with nothing that could ever claim it.
    real_unlink = ct._unlink
    monkeypatch.setattr(
        ct,
        "_unlink",
        lambda path, **kw: None if path == _cert(cf) else real_unlink(path, **kw),
    )
    _provision()
    assert _cert(cf).exists()

    monkeypatch.setattr(ct, "_unlink", real_unlink)
    assert ct.teardown_custom_tunnel() is True
    assert not _cert(cf).exists()
    assert ct._read(ct._RECORD) is None


def test_teardown_removes_only_local_credentials_for_manual_cloudflare_cleanup(cf):
    identity = _provision()
    calls = list(cf.calls)
    assert ct.teardown_custom_tunnel() is True
    assert cf.calls == calls
    assert ct.read_identity() is None
    assert not Path(identity["credentials"]).exists()
    assert ct.orphaned_hostnames() == [_HOST]


def test_a_dns_conflict_is_refused_and_leaves_nothing_to_clean_up(cf):
    cf.route_outcome = (1, "code: 1003, reason: A CNAME record with that host already exists")
    with pytest.raises(ct.ProvisioningError) as excinfo:
        _provision()
    assert excinfo.value.code == "dns_conflict" and _HOST in excinfo.value.detail
    assert not _cert(cf).exists()
    assert cf.deleted == _created(cf)
    assert ct.read_identity() is None
    assert ct._string_list(ct._ORPHANS) == []
    assert ct._read(ct._RECORD) is None
    assert not (ct.state_root() / f"{_TUNNEL_ID}.json").exists()


def test_a_record_created_outside_the_requested_zone_is_refused(cf):
    # Authorizing the wrong zone does not fail the route. cloudflared takes the
    # hostname as a label inside the zone it was given and reports success, so the
    # name it says it created is the only thing that distinguishes the two.
    cf.route_outcome = (0, f"Added CNAME {_HOST}.example.net which will route to this tunnel")
    with pytest.raises(ct.ProvisioningError) as excinfo:
        _provision()
    assert (excinfo.value.code, excinfo.value.detail) == ("hostname_not_authorized", _HOST)
    assert ct.read_identity() is None
    # The route did succeed here, unlike the refusals cloudflared reports itself,
    # so the name it created has to stay accounted for rather than be forgotten.
    # The requested name was never created, so orphaning it would send the user
    # looking for a record that does not exist.
    assert ct.orphaned_hostnames() == [f"{_HOST}.example.net"]
    assert ct._read(ct._RECORD) is None
    assert len(cf.deleted) == 1


@pytest.mark.parametrize(
    "raw",
    # Anything urlsplit would treat as a delimiter returns a shorter name than
    # the one typed, and provisioning that name is worse than refusing it.
    [
        "studio@example.com",
        "example.com:bad",
        "a b.example.com",
        "st#p.example.com",
        "example",
        "",
        # A leading dot shortens the name to the apex domain the user only meant
        # to put a subdomain under.
        ".example.com",
        "example.com..",
        # The stdlib IDNA codec maps this to "fass.de", a separate registration,
        # rather than encoding it. The A-label form below is the way in.
        "studio.faß.de",
    ],
)
def test_a_hostname_that_would_be_truncated_is_refused(raw):
    with pytest.raises(ct.ProvisioningError) as excinfo:
        ct.canonical_hostname(raw)
    assert excinfo.value.code == "invalid_hostname"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Studio.Example.COM.", "studio.example.com"),
        ("https://studio.example.com/path", "studio.example.com"),
        ("  example.com  ", "example.com"),
        ("bücher.example.com", "xn--bcher-kva.example.com"),
        ("日本.example.com", "xn--wgv71a.example.com"),
        # The escape hatch for a name the codec would map away.
        ("studio.xn--fa-hia.de", "studio.xn--fa-hia.de"),
        ("a-b.example.co.uk", "a-b.example.co.uk"),
    ],
)
def test_a_hostname_a_user_would_enter_still_canonicalizes(raw, expected):
    assert ct.canonical_hostname(raw) == expected


@pytest.mark.parametrize(
    "reported",
    # A trailing root label is the same name, and a version that reports nothing
    # must not be read as a mismatch.
    ["Added CNAME " + _HOST + ". which will route to this tunnel", "", "done"],
)
def test_a_route_that_matches_or_says_nothing_is_accepted(cf, reported):
    cf.route_outcome = (0, reported)
    _provision()
    assert ct.read_identity()["hostname"] == _HOST


@pytest.mark.parametrize(
    "message",
    [
        "Failed to add route: code: 7003, reason: Failed to find zone",
        "API request failed: zone could not be found",
    ],
)
def test_route_rejects_authorization_for_a_different_domain(cf, message):
    cf.route_outcome = (1, message)
    with pytest.raises(ct.ProvisioningError) as excinfo:
        _provision()
    assert (excinfo.value.code, excinfo.value.detail) == ("hostname_not_authorized", _HOST)
    assert ct.read_identity() is None
    assert ct.orphaned_hostnames() == []


@pytest.mark.parametrize("cause", [_API_REFUSAL, "Cannot add route: " + "detail " * 50, "", None])
def test_a_route_failure_is_reported_as_itself_and_keeps_the_hostname_owned(cf, cause):
    cf.route_outcome = (1, f"INF Request failed\n{_WRAP}{cause}" if cause is not None else _OFFLINE)
    with pytest.raises(ct.ProvisioningError) as excinfo:
        _provision()
    assert excinfo.value.code == "route_failed"
    assert excinfo.value.detail == (cause if cause is not None else "dial tcp: i/o timeout")[:300]
    assert "--overwrite-dns" not in excinfo.value.detail
    assert ct._string_list(ct._ORPHANS) == [_HOST]


@pytest.mark.parametrize(
    "fails, made, ours", [([_COLLIDED], 2, 1), ([_COLLIDED] * 2, 2, 0), ([_OFFLINE], 1, 1)]
)
def test_only_a_name_collision_regenerates_and_disowns_the_colliding_name(cf, fails, made, ours):
    cf.create_outcomes = [(1, text) for text in fails]
    cf.route_outcome = (1, _API_REFUSAL)
    with pytest.raises(ct.ProvisioningError):
        _provision()
    assert len(set(_created(cf))) == made
    assert cf.deleted == _created(cf)[made - ours :]


@pytest.mark.parametrize("delete_says, still_owed", [(_OFFLINE, True), (_ABSENT, False)])
def test_only_a_tunnel_that_might_still_exist_is_written_down(cf, delete_says, still_owed):
    cf.route_outcome = (1, _API_REFUSAL)
    cf.delete_outcome = (1, delete_says)
    with pytest.raises(ct.ProvisioningError):
        _provision()
    assert ct._string_list(ct._ABANDONED) == (_created(cf) if still_owed else [])
    assert ct._read(ct._RECORD) is None and cf.cert_at_delete == [True]
    assert ct._delete_tunnel(None, "unsloth-AB12CD") is False
    assert ct._delete_tunnel(None, "prod-gateway") is True


def test_a_pre_existing_certificate_is_refused_and_left_alone(cf):
    _cert(cf).parent.mkdir(parents = True, exist_ok = True)
    _cert(cf).write_text("USER-CERT", encoding = "utf-8")
    with pytest.raises(ct.ProvisioningError) as excinfo:
        _provision()
    assert excinfo.value.code == "certificate_exists"
    assert _cert(cf).read_text(encoding = "utf-8") == "USER-CERT"
    assert ct._read(ct._RECORD) is None
    assert cf.calls == []


@pytest.mark.parametrize("found", ["/etc/cloudflared/cert.pem", "/Users/A B/cert.pem", None])
def test_a_certificate_that_appeared_during_login_is_not_adopted(cf, found):
    what = f"an existing certificate at {found}" if found else "a certificate"
    said = f"You have {what} which login would overwrite.\n"
    cf.login = lambda: FakeLogin(_cert(cf), extra = said)
    with pytest.raises(ct.ProvisioningError) as excinfo:
        _provision()
    assert excinfo.value.code == "certificate_exists"
    assert excinfo.value.detail == (found or str(ct.origin_cert_path()))
    assert _cert(cf).read_text(encoding = "utf-8") == "STUDIO-CERT"


@pytest.mark.parametrize("ending", ["cancelled", "login_timed_out", "raises"])
def test_a_login_ended_early_leaves_no_certificate_behind(cf, monkeypatch, ending):
    def explode():
        raise RuntimeError("the flag this check read is gone")

    if ending == "login_timed_out":
        monkeypatch.setattr(ct, "_LOGIN_DEADLINE", 0.0)
    cf.login = lambda: FakeLogin(_cert(cf), alive = 3)
    stop = explode if ending == "raises" else (lambda: True) if ending == "cancelled" else None
    with pytest.raises(RuntimeError if ending == "raises" else ct.ProvisioningError) as excinfo:
        _provision(cancelled = stop)
    assert ending == "raises" or excinfo.value.code == ending
    assert cf.child.returncode == -15
    assert not _cert(cf).exists()
    assert ct._read(ct._RECORD) is None


@pytest.mark.parametrize("cancel_after", ["login", "create", "route"])
def test_a_cancel_arriving_mid_setup_leaves_nothing_owned(cf, monkeypatch, cancel_after):
    # Cancel used to be read once, after login, so one arriving during either of
    # the CLI calls that follow still created a tunnel and a DNS record.
    # Each case cancels as one step returns, so it lands on the gate guarding the
    # next one and no two cases exercise the same gate.
    done = []
    for step, target in (
        ("login", "_run_login"),
        ("create", "_create_tunnel"),
        ("route", "_route_hostname"),
    ):
        original = getattr(ct, target)

        def wrapped(
            *args,
            _step = step,
            _original = original,
            **kwargs,
        ):
            result = _original(*args, **kwargs)
            done.append(_step)
            return result

        monkeypatch.setattr(ct, target, wrapped)

    with pytest.raises(ct.ProvisioningError) as excinfo:
        _provision(cancelled = lambda: cancel_after in done)
    assert excinfo.value.code == "cancelled"
    assert ct.read_identity() is None
    assert ct._read(ct._RECORD) is None
    # Whatever the run got as far as creating is deleted, and a DNS record it
    # created cannot be, so that name is recorded instead of being forgotten.
    assert cf.deleted == ([] if cancel_after == "login" else _created(cf))
    assert ct.orphaned_hostnames() == ([_HOST] if cancel_after == "route" else [])


def test_a_certificate_that_changed_since_it_was_recorded_is_left_alone(cf):
    ct._write(ct._RECORD, {"hostname": _HOST, "cert_digest": "0" * 64})
    _cert(cf).parent.mkdir(parents = True, exist_ok = True)
    _cert(cf).write_text("USER-CERT", encoding = "utf-8")
    _settle()
    assert _cert(cf).read_text(encoding = "utf-8") == "USER-CERT"


def test_the_orphan_is_recorded_before_the_run_is_discarded(cf):
    ct._write(
        ct._RECORD,
        {
            "hostname": _HOST,
            "route_attempted": True,
            "tunnel_names": ["prod-gateway", "unsloth-AB12CD"],
        },
    )
    _settle()
    assert ct._string_list(ct._ORPHANS) == [_HOST]
    assert cf.deleted == ["unsloth-AB12CD"]
    assert ct._read(ct._RECORD) is None


def test_a_held_claim_blocks_every_other_operation_and_is_released_after(cf):
    with ct.certificate_state_claim("teardown"):
        with pytest.raises(ct.ProvisioningError) as excinfo:
            with ct.certificate_state_claim("setup"):
                pass
        assert excinfo.value.code == "certificate_state_busy"
        assert "teardown" in excinfo.value.detail
        with pytest.raises(ct.ProvisioningError) as excinfo:
            _provision()
        assert excinfo.value.code == "certificate_state_busy"
    with ct.certificate_state_claim("teardown"):
        pass


def test_a_claim_held_by_another_process_blocks_this_one(cf, tmp_path):
    path = ct._claim_path()
    ct._writable_dir(path.parent)
    program = (
        "import os,sys,time;"
        f"sys.path.insert(0, {str(ct.Path(ct.__file__).parent)!r});"
        "import cloudflare_tunnel as ct;"
        f"ct._claim_path = lambda: ct.Path({str(path)!r});"
        "ctx = ct.certificate_state_claim('setup');"
        "ctx.__enter__();"
        "print('held', flush = True);"
        "time.sleep(30)"
    )
    holder = subprocess.Popen([sys.executable, "-c", program], stdout = subprocess.PIPE, text = True)
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(ct.ProvisioningError) as excinfo:
            with ct.certificate_state_claim("cleanup"):
                pass
        assert excinfo.value.code == "certificate_state_busy"
        holder.kill()
        holder.wait(timeout = 5)
        with ct.certificate_state_claim("cleanup"):
            pass
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout = 5)


def test_the_login_child_is_written_down_on_disk_while_it_runs(cf):
    _provision()
    seen = cf.child.records[0]
    assert seen["login_pid"] == 999_000 and seen["login_token"]
    assert seen["login_token"] == cf.spawned_with
    assert ct._read(ct._RECORD) is None


def test_a_login_that_ignores_termination_stays_named_in_the_record(cf, monkeypatch):
    # Terminating is best effort, so cancelling a login that ignores it must not
    # clear the pid and token: that is the same unowned certificate again, only
    # reached before cleanup rather than during it.
    class Unkillable(FakeLogin):
        def poll(self):
            return None

        def terminate(self):
            pass

        def wait(self, timeout = None):
            return None

    cf.login = lambda: Unkillable(None)
    monkeypatch.setattr(ct, "_same_process", lambda _token, _pid: True)
    monkeypatch.setattr(ct, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(ct.os, "kill", lambda *_a: None)

    with pytest.raises(ct.ProvisioningError) as excinfo:
        _provision(cancelled = lambda: True)
    assert excinfo.value.code == "cancelled"
    record = ct._read(ct._RECORD)
    assert (record["login_pid"], record["login_token"]) == (999_000, cf.spawned_with)


def test_a_login_child_that_will_not_die_keeps_the_record_that_names_it(cf, monkeypatch):
    # The record holds the pid and token, so discarding it while that login can
    # still write cert.pem leaves a certificate nothing can be shown to own --
    # which the deletion rule fails closed on, permanently.
    ct._write(ct._RECORD, {"hostname": _HOST, "login_pid": 999_000, "login_token": "t"})
    monkeypatch.setattr(ct, "_same_process", lambda _token, _pid: True)
    monkeypatch.setattr(ct, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(ct.os, "kill", lambda *_a: None)

    assert _settle() is False
    assert ct._read(ct._RECORD) == {"hostname": _HOST, "login_pid": 999_000, "login_token": "t"}

    # Once it is gone the same record settles normally.
    monkeypatch.setattr(ct, "_pid_alive", lambda _pid: False)
    _settle()
    assert ct._read(ct._RECORD) is None


@pytest.mark.parametrize("landing", ["adopting the pid", "starting the reader"])
def test_an_interrupt_before_the_wait_still_ends_the_login_child(cf, monkeypatch, landing):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("a shutdown lands with the child already spawned")

    if landing == "adopting the pid":
        monkeypatch.setattr(process_lifetime, "adopt_pid", interrupt)
    else:
        real_thread = threading.Thread

        def only_the_login_reader(*args, **kwargs):
            if kwargs.get("name") == "cloudflared-login":
                interrupt()
            return real_thread(*args, **kwargs)

        monkeypatch.setattr(ct.threading, "Thread", only_the_login_reader)
    with pytest.raises(KeyboardInterrupt):
        _provision()
    assert cf.child is not None and cf.child.returncode == -15
    assert 999_000 not in process_lifetime._tracked_pids


def test_every_command_is_pinned_to_the_certificate_we_proved(monkeypatch):
    monkeypatch.setenv("TUNNEL_ORIGIN_CERT", "/somewhere/else/cert.pem")
    monkeypatch.setenv("TUNNEL_FORCE_PROVISIONING_DNS", "1")
    sent = {}
    monkeypatch.setattr(ct.subprocess, "run", lambda a, **k: sent.update(argv = a, env = k["env"]))
    ct._cli("cloudflared", "route", "dns", "unsloth-AB12CD", _HOST)
    argv = sent["argv"]
    assert argv[argv.index("--origincert") + 1] == str(ct.origin_cert_path())
    assert not {"--overwrite-dns", "-f"} & set(argv)
    assert not any(key.startswith("TUNNEL_") for key in sent["env"])
