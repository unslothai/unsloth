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
import sys
import tarfile
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
    assert cached.stat().st_mode & 0o111  # executable bit set


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

        def read(self, n=-1):
            if self._sent:
                return b""
            self._sent = True
            return b"payload"

    def fake_urlopen(req, timeout=None):
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
    assert cached.stat().st_mode & 0o111  # chmod applied on posix
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


def test_wait_for_url_times_out_without_blocking():
    t = ct.CloudflareTunnel(8080, "/bin/cloudflared")
    assert t.wait_for_url(timeout = 0.05) is None


def test_start_studio_tunnel_no_binary(monkeypatch):
    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: None)
    assert ct.start_studio_tunnel(8080) is None


def test_start_studio_tunnel_returns_url(monkeypatch):
    class _StubTunnel:
        def __init__(self, port, binary):
            self.url = None

        def start(self):
            self.url = "https://stub-xyz.trycloudflare.com"

        def wait_for_url(self, timeout):
            return self.url

        def stop(self):
            pass

    monkeypatch.setattr(ct, "ensure_cloudflared", lambda: "/bin/cloudflared")
    monkeypatch.setattr(ct, "CloudflareTunnel", _StubTunnel)
    try:
        assert ct.start_studio_tunnel(8080) == "https://stub-xyz.trycloudflare.com"
    finally:
        ct.stop_studio_tunnel()


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


def test_run_server_cloudflare_default_true():
    defaults = _func_param_defaults(_RUN_PY.read_text(), "run_server")
    assert defaults.get("cloudflare") is True


def test_argparse_cloudflare_default_true():
    assert _argparse_default(_RUN_PY.read_text(), "--cloudflare") is True


def test_run_server_gates_tunnel_on_wildcard():
    # Guard against accidentally widening the trigger beyond 0.0.0.0.
    source = _RUN_PY.read_text()
    assert "_cloudflare_enabled" in source
    assert 'host == "0.0.0.0"' in source
