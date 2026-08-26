# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where the launcher refresh gets install.sh / install.ps1, and what it refuses to run.

`unsloth studio update` re-runs the installer with --shortcuts-only. It looks in a source
checkout first, so `update --local` tests its own installer; then at unsloth.ai, so a
launcher fix reaches users without waiting for a release; then at the copy that shipped with
this release under <data>/share/unsloth.

Fetching, rather than only shipping a copy, is deliberate, and unsloth.ai stays the source.
What the bundled copy adds is a floor: a fetch that does not land used to mean no launcher
refresh at all, and now means a release-matched one. So these tests pin the source order, the
allowed hosts, the UNSLOTH_NO_REMOTE_INSTALLER=1 opt-out, and everything around the fetch: a
transport error cannot abort an update that already succeeded, a response that is not an
installer is not piped into bash, and a fetch that does not land falls through to disk.
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


def _posix(
    monkeypatch,
    tmp_path,
    bundled = (),
):
    """POSIX refresh with no checkout in sight, the shape of a wheel install. The tests run
    from a clone, so _PACKAGE_ROOT would otherwise short-circuit every fetch, and the
    interpreter running them may have a real <data>/share/unsloth to fall through to.
    """
    studio = _studio()
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    monkeypatch.setattr(studio, "_PACKAGE_ROOT", tmp_path / "no-checkout-here")
    monkeypatch.delenv("STUDIO_LOCAL_REPO", raising = False)
    monkeypatch.delenv("UNSLOTH_NO_REMOTE_INSTALLER", raising = False)
    monkeypatch.setattr(studio, "_bundled_installer_roots", lambda: list(bundled))
    return studio


def _bundled_root(tmp_path, name = "install.sh"):
    """A <data>/share/unsloth holding the installer that shipped with the release."""
    root = tmp_path / "venv" / "share" / "unsloth"
    root.mkdir(parents = True, exist_ok = True)
    (root / name).write_bytes(_SH)
    return root


# ── where the installer comes from ─────────────────────────────────────────────────


def test_the_installer_is_fetched_from_unsloth_ai():
    """unsloth.ai is the documented install URL, and the refresh uses the same one."""
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
        "https://unsloth.ai@evil.example/install.sh",
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


# ── the copy that shipped with the release ─────────────────────────────────────────


def test_an_unreachable_origin_falls_back_to_the_bundled_installer(monkeypatch, tmp_path):
    """An offline machine refreshes its launcher instead of silently skipping one."""
    root = _bundled_root(tmp_path)
    studio = _posix(monkeypatch, tmp_path, bundled = [root])
    monkeypatch.setattr(studio, "_fetch_installer", lambda *a, **k: None)
    runs = []
    monkeypatch.setattr(
        studio.subprocess, "run", lambda argv, **kw: runs.append(list(argv)) or _Result()
    )
    studio._refresh_desktop_shortcuts()
    assert runs == [["bash", str(root / "install.sh"), "--shortcuts-only"]]


def test_a_fetched_installer_that_fails_falls_back_to_the_bundled_one(monkeypatch, tmp_path):
    """A published installer that is broken, or incompatible with the installed release, must
    not consume the fallback: a release-matched copy is sitting on disk."""
    root = _bundled_root(tmp_path)
    studio = _posix(monkeypatch, tmp_path, bundled = [root])
    monkeypatch.setattr(studio, "_fetch_installer", lambda *a, **k: _SH)

    runs = []

    def _run(argv, **kwargs):
        runs.append(list(argv))

        class _R:
            returncode = 1 if list(argv)[:3] == ["bash", "-s", "--"] else 0

        return _R()

    monkeypatch.setattr(studio.subprocess, "run", _run)
    studio._refresh_desktop_shortcuts()
    assert len(runs) == 2, runs
    assert runs[0][:3] == ["bash", "-s", "--"], runs[0]
    assert runs[1] == ["bash", str(root / "install.sh"), "--shortcuts-only"]


def test_no_remote_installer_pins_the_refresh_to_the_bundled_copy(monkeypatch, tmp_path):
    """The opt-out for air-gapped machines, and for anyone who would rather an update
    never ran a freshly fetched script."""
    root = _bundled_root(tmp_path)
    studio = _posix(monkeypatch, tmp_path, bundled = [root])
    monkeypatch.setenv("UNSLOTH_NO_REMOTE_INSTALLER", "1")
    monkeypatch.setattr(
        studio,
        "_fetch_installer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("the opt-out must not fetch")),
    )
    runs = []
    monkeypatch.setattr(
        studio.subprocess, "run", lambda argv, **kw: runs.append(list(argv)) or _Result()
    )
    studio._refresh_desktop_shortcuts()
    assert runs == [["bash", str(root / "install.sh"), "--shortcuts-only"]]


def test_the_managed_venv_leads_the_bundled_lookup(monkeypatch, tmp_path):
    """A pip-installed CLI can drive an update into STUDIO_HOME/unsloth_studio, where setup
    has just written the new data files. Searching its own prefixes first would run that
    foreign CLI's older bundled installer instead."""
    studio = _studio()
    monkeypatch.setattr(studio, "STUDIO_HOME", tmp_path / "studio-home")
    roots = studio._bundled_installer_roots()
    assert roots[0] == tmp_path / "studio-home" / "unsloth_studio" / "share" / "unsloth"
    assert len(roots) == len(set(roots)), roots
    assert all(root.name == "unsloth" and root.parent.name == "share" for root in roots)


def test_the_bundled_lookup_covers_the_layouts_pip_can_produce(monkeypatch, tmp_path):
    """<data> is not always sys.prefix.

    `pip install --target DIR` collapses purelib and data onto DIR, so the installer sits
    next to the packages and _PACKAGE_ROOT is the only root that finds it. Debian and Ubuntu
    patch the default scheme to posix_local, so a system-python install lands under
    /usr/local while sys.prefix stays /usr. `--user` lands under site.USER_BASE, which is
    ~/.local on Linux but ~/Library/Python/X.Y on macOS.
    """
    studio = _studio()
    target = tmp_path / "target"
    monkeypatch.setattr(studio, "_PACKAGE_ROOT", target)
    monkeypatch.setattr(studio.sysconfig, "get_path", lambda name, *a, **k: "/usr/local")
    monkeypatch.setattr(studio.site, "USER_BASE", str(tmp_path / "userbase"))
    roots = studio._bundled_installer_roots()

    for expected in (
        target / "share" / "unsloth",
        Path(sys.prefix) / "share" / "unsloth",
        Path("/usr/local") / "share" / "unsloth",
        tmp_path / "userbase" / "share" / "unsloth",
    ):
        assert expected in roots, f"{expected} missing from {roots}"
    assert len(roots) == len(set(roots)), roots


def test_a_missing_user_base_does_not_break_the_lookup(monkeypatch):
    """site.USER_BASE is None under an interpreter started with -s or embedded."""
    studio = _studio()
    monkeypatch.setattr(studio.site, "USER_BASE", None)
    roots = studio._bundled_installer_roots()
    assert roots and all(isinstance(root, Path) for root in roots)


def test_neither_the_network_nor_a_bundled_copy_executes_nothing(monkeypatch, tmp_path, capsys):
    studio = _posix(monkeypatch, tmp_path)
    monkeypatch.setattr(studio, "_fetch_installer", lambda *a, **k: None)
    monkeypatch.setattr(
        studio.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("nothing to execute")),
    )
    studio._refresh_desktop_shortcuts()
    assert "skipped: no usable install.sh" in capsys.readouterr().out


# ── what a bad response must not do ────────────────────────────────────────────────


def test_a_failed_fetch_skips_instead_of_executing(monkeypatch, tmp_path):
    """A transport failure must not raise out of an update that already succeeded."""
    studio = _posix(monkeypatch, tmp_path)
    monkeypatch.setattr(studio, "_fetch_installer", lambda *a, **k: None)
    monkeypatch.setattr(
        studio.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("nothing to execute")),
    )
    studio._refresh_desktop_shortcuts()


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
    # The tests run from a clone, so _PACKAGE_ROOT would otherwise supply a second
    # usable installer and the network would never be the fallback under test.
    monkeypatch.setattr(studio, "_PACKAGE_ROOT", tmp_path / "no-checkout-here")
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


def test_building_the_opener_never_mutates_global_urllib(monkeypatch):
    """A private opener is needed to check redirects, but it must stay private.

    OpenerDirector.add_handler() assigns handler.parent, so an earlier attempt to carry
    the site-installed handlers over repointed that opener's own handlers at ours and
    broke every later urlopen() in the process.
    """
    studio = _studio()
    import urllib.request

    site = urllib.request.ProxyHandler({"https": "http://proxy.example:3128"})
    installed = urllib.request.build_opener(site)
    monkeypatch.setattr(urllib.request, "_opener", installed, raising = False)

    opener = studio._build_installer_opener()
    assert any(
        isinstance(h, studio._InstallerRedirectHandler) for h in opener.handlers
    ), "redirect validation was lost"
    assert site.parent is installed, "the installed opener's handler was hijacked"
    assert urllib.request._opener is installed


def test_a_second_on_disk_installer_is_tried_before_the_network(monkeypatch, tmp_path):
    """The old loop probed and launched in one pass, so an unlaunchable candidate left
    the next one to try. Selecting only the first match would drop that second chance
    and reach for the network while a usable installer sat on disk."""
    studio = _studio()
    local = tmp_path / "local"
    pkg = tmp_path / "pkg"
    for d in (local, pkg):
        d.mkdir()
        (d / "install.sh").write_text("#!/bin/sh\n")
    monkeypatch.setenv("STUDIO_LOCAL_REPO", str(local))
    monkeypatch.setattr(studio, "_PACKAGE_ROOT", pkg)
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")

    seen = []

    def _run(argv, **kwargs):
        seen.append(list(argv))
        if str(local / "install.sh") in argv:
            raise OSError(11, "Resource temporarily unavailable")

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(studio.subprocess, "run", _run)
    monkeypatch.setattr(
        studio,
        "_fetch_installer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("a usable installer was on disk")),
    )
    studio._refresh_desktop_shortcuts()

    assert len(seen) == 2, seen
    assert str(local / "install.sh") in seen[0]
    assert str(pkg / "install.sh") in seen[1]
