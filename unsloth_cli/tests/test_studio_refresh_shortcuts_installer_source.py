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

import os
import sys
import tempfile
from pathlib import Path

import pytest

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


def test_a_truncated_body_is_never_executed(monkeypatch, tmp_path):
    """read(amt) does not check Content-Length, so a cut-off transfer must be caught.

    The markers sit early in install.sh, so a body truncated a third of the way in
    still satisfies _looks_like_installer. Without the follow-up read() that forces
    http.client to compare against the declared length, that half-written script
    would be piped into bash.
    """
    import http.client

    studio = _posix(monkeypatch, tmp_path)
    full = _SH + b"create_studio_shortcuts --shortcuts-only\n" + b"# tail\n" * 500
    short = full[: len(full) // 3]
    assert studio._looks_like_installer(
        short, "install.sh"
    ), "the truncated prefix still looks valid"

    class _Response:
        def __init__(self):
            self.calls = 0

        def read(self, n = None):
            self.calls += 1
            if self.calls == 1:
                return short
            # What http.client raises when the body is shorter than Content-Length.
            raise http.client.IncompleteRead(b"", len(full) - len(short))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Opener:
        def open(self, *a, **k):
            return _Response()

    monkeypatch.setattr(studio.urllib.request, "build_opener", lambda *a, **k: _Opener())
    assert studio._fetch_installer("install.sh") is None


def test_a_complete_body_survives_the_completeness_check(monkeypatch, tmp_path):
    """The follow-up read() must not break the normal case: b"" means we have it all."""
    studio = _posix(monkeypatch, tmp_path)

    class _Response:
        def __init__(self):
            self.calls = 0

        def read(self, n = None):
            self.calls += 1
            return _SH if self.calls == 1 else b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Opener:
        def open(self, *a, **k):
            return _Response()

    monkeypatch.setattr(studio.urllib.request, "build_opener", lambda *a, **k: _Opener())
    assert studio._fetch_installer("install.sh") == _SH


def test_a_windows_tempfile_failure_skips_instead_of_aborting(monkeypatch, tmp_path):
    """The refresh runs after the package update has already succeeded.

    A full disk, a read-only %TEMP% or AV holding the handle raises OSError while
    creating or writing the script. That must not surface as a traceback from a
    command whose real work is done.
    """
    studio = _studio()
    # Captured BEFORE any patching: studio.tempfile is the tempfile module itself, so
    # patching through it replaces the global.
    real_mkstemp = tempfile.mkstemp
    calls = []

    def _boom_mkstemp(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(studio.tempfile, "mkstemp", _boom_mkstemp)
    monkeypatch.setattr(
        studio.subprocess,
        "run",
        lambda *a, **k: calls.append(a) or (_ for _ in ()).throw(AssertionError("must not run")),
    )
    studio._run_fetched_installer_ps1(b"x", ["--shortcuts-only"], ["powershell.exe"], {})
    assert calls == []

    # And a write that fails after the file exists still cleans up and skips.
    made = {}

    def _ok_mkstemp(*a, **k):
        fd, path = real_mkstemp(dir = tmp_path)
        made["path"] = path
        made["fd"] = fd
        return fd, path

    class _BadHandle:
        """Stands in for the real handle, and owns the descriptor like one.

        Windows refuses to unlink a file that still has an open descriptor, so a
        handle that leaked the fd would make the cleanup below fail there for a
        reason that has nothing to do with the code under test.
        """

        def __init__(self, fd):
            self._fd = fd

        def __enter__(self):
            return self

        def __exit__(self, *a):
            os.close(self._fd)
            return False

        def write(self, _data):
            raise OSError(5, "Input/output error")

    monkeypatch.setattr(studio.tempfile, "mkstemp", _ok_mkstemp)
    monkeypatch.setattr(studio.os, "fdopen", lambda fd, *a, **k: _BadHandle(fd))
    studio._run_fetched_installer_ps1(b"x", ["--shortcuts-only"], ["powershell.exe"], {})
    assert not Path(made["path"]).exists(), "the temp script must not be left behind"
    # Asserted directly so the descriptor leak is caught on Linux too, rather than
    # only surfacing as an unlink failure on Windows.
    with pytest.raises(OSError):
        os.fstat(made["fd"])


def test_a_powershell_help_block_is_not_mistaken_for_html():
    """`<#` opens comment-based help. Rejecting it would kill every refresh silently."""
    studio = _studio()
    ps1 = (
        b"<#\n.SYNOPSIS\nUnsloth Studio installer\n#>\n"
        b"function Install-UnslothStudio { }\n--shortcuts-only\n" + b"# pad\n" * 200
    )
    assert studio._looks_like_installer(ps1, "install.ps1")


def test_real_html_is_still_rejected():
    studio = _studio()
    for body in (
        b"<!DOCTYPE html><html><body>proxy error</body></html>" + b"--shortcuts-only\n" * 200,
        b"<html><head><title>502</title></head></html>" + b"--shortcuts-only\n" * 200,
        b"<?xml version='1.0'?><error/>" + b"--shortcuts-only\n" * 200,
    ):
        assert not studio._looks_like_installer(body, "install.sh"), body[:30]


def test_markers_do_not_pin_internal_function_names():
    """A renamed internal helper must not disable refreshes for everyone."""
    studio = _studio()
    renamed = b"#!/bin/sh\nmake_studio_launchers() { :; }\n--shortcuts-only\n" + b"# pad\n" * 200
    assert studio._looks_like_installer(renamed, "install.sh")
    for markers in studio._INSTALLER_MARKERS.values():
        assert markers == (b"--shortcuts-only",)


def test_a_local_installer_that_cannot_be_launched_falls_back_to_the_network(monkeypatch, tmp_path):
    """Pre-refactor the candidate loop caught the exec OSError and carried on.

    A machine that cannot spawn bash for the checkout must still reach the fetch, or a
    refactor that only reorganised the code would have stopped refreshing launchers.
    """
    studio = _studio()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "install.sh").write_text("#!/bin/sh\n")
    monkeypatch.setenv("STUDIO_LOCAL_REPO", str(checkout))
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")

    seen = []

    def _run(argv, **kwargs):
        seen.append(list(argv))
        if str(checkout / "install.sh") in argv:
            raise OSError(8, "Exec format error")

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(studio.subprocess, "run", _run)
    monkeypatch.setattr(studio, "_fetch_installer", lambda *a, **k: _SH)
    studio._refresh_desktop_shortcuts()

    assert len(seen) == 2, seen
    assert str(checkout / "install.sh") in seen[0]
    assert seen[1][:3] == ["bash", "-s", "--"], seen[1]


def test_a_site_installed_opener_is_still_honoured(monkeypatch):
    """urlopen() used whatever install_opener() set. Dropping it would strand any
    machine whose proxy auth or corporate CA lives in a site-wide handler."""
    studio = _studio()
    import urllib.request

    # A handler with no recognised *_open/*_request method is never registered by
    # add_handler at all, so use the shape a corporate site actually installs.
    marker = urllib.request.ProxyHandler({"https": "http://proxy.example:3128"})
    installed = urllib.request.build_opener(marker)
    monkeypatch.setattr(urllib.request, "_opener", installed, raising = False)

    opener = studio._build_installer_opener()
    assert any(h is marker for h in opener.handlers), "site handler was dropped"
    assert any(
        isinstance(h, studio._InstallerRedirectHandler) for h in opener.handlers
    ), "redirect validation was lost"

    # A private attribute of an unexpected shape must not cost anyone a refresh.
    monkeypatch.setattr(urllib.request, "_opener", object(), raising = False)
    fallback = studio._build_installer_opener()
    assert any(isinstance(h, studio._InstallerRedirectHandler) for h in fallback.handlers)
