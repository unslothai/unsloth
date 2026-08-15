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
