# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression for PR #6295: the banner's canned http://127.0.0.1 URL is valid
only for the exact loopback aliases, so any other bind (e.g. a specific LAN IP)
must show its real address."""

import io
import sys

import pytest

from startup_banner import print_studio_access_banner


def test_non_alias_loopback_shows_real_address(capsys):
    # A server bound to 127.0.0.2 does not listen on 127.0.0.1.
    print_studio_access_banner(port = 8891, bind_host = "127.0.0.2", display_host = "127.0.0.2")
    out = capsys.readouterr().out
    assert "http://127.0.0.2:8891" in out
    assert "http://127.0.0.1" not in out


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost"])
def test_alias_loopback_shows_canned_url(capsys, host):
    print_studio_access_banner(port = 8891, bind_host = host, display_host = host)
    assert "http://127.0.0.1:8891" in capsys.readouterr().out


@pytest.mark.parametrize(
    "host,loopback_url",
    [
        ("::0", "http://[::1]:8891"),
        ("0:0:0:0:0:0:0:0", "http://[::1]:8891"),
        ("0", "http://127.0.0.1:8891"),
        ("::ffff:0.0.0.0", "http://127.0.0.1:8891"),
    ],
)
def test_wildcard_aliases_show_reachable_urls(capsys, host, loopback_url):
    print_studio_access_banner(port = 8891, bind_host = host, display_host = "192.168.1.24")
    out = capsys.readouterr().out
    assert loopback_url in out
    assert "http://192.168.1.24:8891" in out


WSL_HINT = "WSL2: open http://localhost:"


@pytest.mark.parametrize("mode,wsl", [("nat", True), (None, False)])
def test_wsl_hint_replaces_the_private_address_note(capsys, monkeypatch, mode, wsl):
    import lan_access
    import run
    from utils.paths import file_manager

    monkeypatch.setattr(lan_access, "_wsl_networking_mode", lambda: mode)
    monkeypatch.setattr(file_manager, "_in_container", lambda: False)
    monkeypatch.setattr(run, "_network_share_host_for_bind", lambda h: h)
    monkeypatch.setattr(run, "_print_cloudflare_line", lambda *a, **k: None)
    monkeypatch.setattr(run, "_localhost_ipv6_mismatch_url", lambda *a, **k: None)
    run._emit_startup_output("0.0.0.0", 8888, "172.25.35.232")
    out = capsys.readouterr().out
    assert (WSL_HINT + "8888" in out) is wsl
    assert ("networkingMode=mirrored" in out) is wsl
    assert ("172.25.35.232 is a private/LAN address" in out) is not wsl
    assert run._public_reachable is False
    if wsl:
        assert out.index("/api/health") < out.index(WSL_HINT)


@pytest.mark.parametrize(
    "host,mode,expected",
    [
        ("0.0.0.0", "nat", True),
        ("::", "nat", True),
        ("0.0.0.0", "unknown", True),
        ("0.0.0.0", "none", False),
        ("0.0.0.0", "mirrored", False),
        ("0.0.0.0", None, False),
        ("127.0.0.1", "nat", False),
    ],
)
def test_startup_output_wsl_hint_gating(capsys, monkeypatch, host, mode, expected):
    import lan_access
    import run
    from utils.paths import file_manager

    monkeypatch.setattr(lan_access, "_wsl_networking_mode", lambda: mode)
    monkeypatch.setattr(file_manager, "_in_container", lambda: False)
    monkeypatch.setattr(run, "_network_share_host_for_bind", lambda h: h)
    monkeypatch.setattr(run, "_verify_global_reachability", lambda *a, **k: None)
    monkeypatch.setattr(run, "_print_cloudflare_line", lambda *a, **k: None)
    monkeypatch.setattr(run, "_localhost_ipv6_mismatch_url", lambda *a, **k: None)
    run._emit_startup_output(host, 8888, host)
    assert (WSL_HINT in capsys.readouterr().out) is expected


def test_startup_output_wsl_hint_skipped_in_container(capsys, monkeypatch):
    import lan_access
    import run
    from utils.paths import file_manager

    monkeypatch.setattr(lan_access, "_wsl_networking_mode", lambda: "unknown")
    monkeypatch.setattr(file_manager, "_in_container", lambda: True)
    monkeypatch.setattr(run, "_network_share_host_for_bind", lambda h: h)
    monkeypatch.setattr(run, "_verify_global_reachability", lambda *a, **k: None)
    monkeypatch.setattr(run, "_print_cloudflare_line", lambda *a, **k: None)
    monkeypatch.setattr(run, "_localhost_ipv6_mismatch_url", lambda *a, **k: None)
    run._emit_startup_output("0.0.0.0", 8000, "0.0.0.0")
    assert WSL_HINT not in capsys.readouterr().out


def test_wsl_hint_check_skips_container_import_off_wsl(capsys, monkeypatch):
    # tests/studio/install stubs utils.paths without file_manager.
    import lan_access
    import run

    monkeypatch.setattr(lan_access, "_wsl_networking_mode", lambda: None)
    monkeypatch.setitem(sys.modules, "utils.paths.file_manager", None)
    monkeypatch.setattr(run, "_network_share_host_for_bind", lambda h: h)
    monkeypatch.setattr(run, "_verify_global_reachability", lambda *a, **k: None)
    monkeypatch.setattr(run, "_print_cloudflare_line", lambda *a, **k: None)
    monkeypatch.setattr(run, "_localhost_ipv6_mismatch_url", lambda *a, **k: None)
    run._emit_startup_output("0.0.0.0", 8000, "0.0.0.0")
    assert WSL_HINT not in capsys.readouterr().out


def test_banner_prints_on_strict_cp1252_stdout(monkeypatch):
    buf = io.BytesIO()
    stdout = io.TextIOWrapper(buf, encoding = "cp1252", errors = "strict")
    monkeypatch.setattr(sys, "stdout", stdout)

    print_studio_access_banner(port = 8891, bind_host = "127.0.0.1", display_host = "127.0.0.1")
    stdout.flush()

    out = buf.getvalue().decode("cp1252")
    assert "? Unsloth Studio is running" in out


def test_banner_print_fallback_handles_unknown_stdout_encoding(monkeypatch):
    class InvalidEncodingStdout:
        encoding = "not-a-real-codec"

        def __init__(self):
            self.buf = io.BytesIO()
            self.inner = io.TextIOWrapper(self.buf, encoding = "cp1252", errors = "strict")

        def write(self, text):
            return self.inner.write(text)

        def flush(self):
            return self.inner.flush()

        def getvalue(self):
            self.flush()
            return self.buf.getvalue().decode("cp1252")

    stdout = InvalidEncodingStdout()
    monkeypatch.setattr(sys, "stdout", stdout)

    print_studio_access_banner(port = 8891, bind_host = "127.0.0.1", display_host = "127.0.0.1")

    assert "? Unsloth Studio is running" in stdout.getvalue()
