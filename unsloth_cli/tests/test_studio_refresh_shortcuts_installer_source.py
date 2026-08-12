# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The launcher refresh fetches install.sh / install.ps1 from unsloth.ai and runs it.

`unsloth studio update` re-runs the installer with --shortcuts-only. Fetching, rather
than shipping a copy in the wheel, is deliberate: a launcher fix then reaches users
without waiting for a release. unsloth.ai and the unslothai/unsloth repo it redirects to
are trusted, so what is left to get right is everything around the fetch, namely that a
transport error cannot abort an update that already succeeded, that a response which is
not an installer is not piped into bash, and that a source checkout still outranks the
network so `update --local` tests its own installer.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SH = b"#!/bin/sh\ncreate_studio_shortcuts() { :; }\n--shortcuts-only\n" + b"# pad\n" * 200


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


class _Result:
    returncode = 0


def _posix(monkeypatch, tmp_path):
    """POSIX refresh with no checkout in sight, the shape of a wheel install. The tests run
    from a clone, so _PACKAGE_ROOT would otherwise short-circuit every fetch.
    """
    studio = _studio()
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    monkeypatch.setattr(studio, "_PACKAGE_ROOT", tmp_path / "no-checkout-here")
    monkeypatch.delenv("STUDIO_LOCAL_REPO", raising = False)
    return studio


# ── where the installer comes from ─────────────────────────────────────────────────


def test_the_installer_is_fetched_from_unsloth_ai():
    studio = _studio()
    assert studio._INSTALLER_URL_BASH == "https://unsloth.ai/install.sh"
    assert studio._INSTALLER_URL_PWSH == "https://unsloth.ai/install.ps1"


def test_the_redirect_chain_is_allowed_and_nothing_else():
    """unsloth.ai 301s to raw.githubusercontent, so both are in the chain."""
    studio = _studio()
    for good in (
        "https://unsloth.ai/install.sh",
        "https://raw.githubusercontent.com/unslothai/unsloth/main/install.sh",
    ):
        assert studio._is_allowed_installer_url(good), good
    for bad in (
        "https://evil.example/install.sh",
        "http://unsloth.ai/install.sh",
        "https://unsloth.ai.evil.example/install.sh",
    ):
        assert not studio._is_allowed_installer_url(bad), bad


def test_a_redirect_off_the_chain_is_refused():
    import urllib.error

    studio = _studio()
    handler = studio._InstallerRedirectHandler()
    try:
        handler.redirect_request(None, None, 302, "Found", {}, "https://evil.example/x.sh")
    except urllib.error.URLError:
        return
    raise AssertionError("followed a redirect off the allowed chain")


def test_a_checkout_outranks_the_network(monkeypatch, tmp_path):
    """`update --local` is testing its own installer, so no fetch may happen."""
    studio = _posix(monkeypatch, tmp_path)
    checkout = tmp_path / "install.sh"
    checkout.write_bytes(_SH)
    monkeypatch.setenv("STUDIO_LOCAL_REPO", str(tmp_path))
    monkeypatch.setattr(
        studio,
        "_fetch_installer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("checkout must not fetch")),
    )
    runs = []
    monkeypatch.setattr(studio.subprocess, "run", lambda argv, **kw: runs.append(argv) or _Result())
    studio._refresh_desktop_shortcuts()
    assert runs == [["bash", str(checkout), "--shortcuts-only"]]


def test_a_wheel_install_fetches_and_pipes_to_bash(monkeypatch, tmp_path):
    studio = _posix(monkeypatch, tmp_path)
    monkeypatch.setattr(studio, "_fetch_installer", lambda *a, **k: b"FETCHED")
    runs = []
    monkeypatch.setattr(
        studio.subprocess,
        "run",
        lambda argv, **kw: runs.append((argv, kw.get("input"))) or _Result(),
    )
    studio._refresh_desktop_shortcuts()
    assert runs == [(["bash", "-s", "--", "--shortcuts-only"], b"FETCHED")]


# ── what a bad response must not do ────────────────────────────────────────────────


def test_a_failed_fetch_skips_instead_of_executing(monkeypatch, tmp_path, capsys):
    studio = _posix(monkeypatch, tmp_path)
    monkeypatch.setattr(studio, "_fetch_installer", lambda *a, **k: None)
    monkeypatch.setattr(
        studio.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("nothing to execute")),
    )
    studio._refresh_desktop_shortcuts()
    assert capsys.readouterr().out == ""


def test_non_installer_responses_are_not_executed():
    """A captive portal or an error body must never be piped into bash."""
    studio = _studio()
    for body in (
        None,
        b"",
        b"<!DOCTYPE html><html><body>Sign in to WiFi</body></html>" + b"x" * 4000,
        b"  <html>" + b"x" * 4000,
        b"rm -rf ~" + b"\n" * 4000,
        _SH[:200],
    ):
        assert not studio._looks_like_installer(body, "install.sh")
    assert studio._looks_like_installer(_SH, "install.sh")


def test_http_framing_errors_are_fetch_failures_not_crashes(monkeypatch, tmp_path):
    """IncompleteRead is neither URLError nor OSError, and must not abort the update."""
    import http.client

    studio = _posix(monkeypatch, tmp_path)
    assert not issubclass(http.client.IncompleteRead, (OSError, ValueError))
    for exc in (
        http.client.IncompleteRead(b"half"),
        http.client.BadStatusLine("nonsense"),
        http.client.LineTooLong("header line"),
    ):

        class _Opener:
            def open(self, *a, **k):
                raise exc

        monkeypatch.setattr(studio.urllib.request, "build_opener", lambda *a, **k: _Opener())
        assert studio._fetch_installer("install.sh") is None, exc


def test_oversized_and_html_responses_return_none(monkeypatch, tmp_path):
    studio = _posix(monkeypatch, tmp_path)

    class _Response:
        def __init__(self, body):
            self._body = body

        def read(self, n = None):
            return self._body[:n] if n else self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    for body in (
        b"x" * (studio._INSTALLER_MAX_BYTES + 1),
        b"<html>not an installer</html>" + b"x" * 4000,
    ):

        class _Opener:
            def open(self, *a, **k):
                return _Response(body)

        monkeypatch.setattr(studio.urllib.request, "build_opener", lambda *a, **k: _Opener())
        assert studio._fetch_installer("install.sh") is None
