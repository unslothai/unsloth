# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Launcher refresh: fetch install.sh / install.ps1 from main, bundled copy as fallback.

`unsloth studio update` re-runs the installer with --shortcuts-only. It used to fetch
https://unsloth.ai/install.sh and pipe it into `bash -s`, with no local copy to fall
back to: a PyPI install has no installer next to the package, so a failed or hijacked
fetch was the difference between refreshing a launcher and running someone else's shell
script. Two things changed:

  * the fetch goes to raw.githubusercontent.com, the origin the unsloth.ai URL was a
    Cloudflare 301 to, so the redirect's control plane is no longer in the path
  * pyproject ships the installers under <data>/share/unsloth, so there is always a
    local installer to fall back to when the fetch fails or returns something that is
    not an installer

A source checkout still outranks both: `update --local` must test its own installer.
"""

from __future__ import annotations

import sys
import sysconfig
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
    """POSIX refresh with no checkout in sight, which is the shape of a wheel install.

    The tests run from a clone, so _PACKAGE_ROOT would otherwise find the repo's own
    install.sh and short-circuit every fetch.
    """
    studio = _studio()
    monkeypatch.setattr(studio.platform, "system", lambda: "Linux")
    monkeypatch.setattr(studio, "_PACKAGE_ROOT", tmp_path / "no-checkout-here")
    monkeypatch.delenv("UNSLOTH_NO_REMOTE_INSTALLER", raising = False)
    monkeypatch.delenv("STUDIO_LOCAL_REPO", raising = False)
    return studio


def _record_runs(monkeypatch, studio):
    runs = []

    def _run(argv, **kwargs):
        runs.append((argv, kwargs.get("input")))
        return _Result()

    monkeypatch.setattr(studio.subprocess, "run", _run)
    return runs


# ── source of the fetch ────────────────────────────────────────────────────────────


def test_fetch_targets_the_repo_origin_not_the_website():
    studio = _studio()
    for url in (studio._INSTALLER_URL_BASH, studio._INSTALLER_URL_PWSH):
        assert url.startswith("https://raw.githubusercontent.com/unslothai/unsloth/")
    assert "unsloth.ai/install" not in studio._INSTALLER_URL_BASH


def test_redirect_off_the_pinned_host_is_refused():
    studio = _studio()
    handler = studio._InstallerRedirectHandler()
    import urllib.error

    for bad in ("https://evil.example/install.sh", "http://raw.githubusercontent.com/x"):
        try:
            handler.redirect_request(None, None, 302, "Found", {}, bad)
        except urllib.error.URLError:
            continue
        raise AssertionError(f"followed redirect to {bad}")


# ── candidate resolution ───────────────────────────────────────────────────────────


def test_bundled_candidate_covers_wheel_installs():
    studio = _studio()
    shipped = Path(sysconfig.get_path("data")) / "share" / "unsloth" / "install.sh"
    assert shipped in studio._installer_bundled_candidates("install.sh")


def test_checkout_outranks_everything(monkeypatch, tmp_path):
    """A --local update tests its own installer, so no fetch may happen."""
    studio = _posix(monkeypatch, tmp_path)
    checkout = tmp_path / "install.sh"
    checkout.write_bytes(_SH)
    monkeypatch.setenv("STUDIO_LOCAL_REPO", str(tmp_path))
    monkeypatch.setattr(
        studio,
        "_fetch_installer",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("checkout must not fetch")),
    )
    runs = _record_runs(monkeypatch, studio)
    studio._refresh_desktop_shortcuts()
    assert runs == [(["bash", str(checkout), "--shortcuts-only"], None)]


# ── fetch first, bundled fallback ──────────────────────────────────────────────────


def test_fetched_installer_is_preferred_over_the_bundled_copy(monkeypatch, tmp_path):
    studio = _posix(monkeypatch, tmp_path)
    bundled = tmp_path / "install.sh"
    bundled.write_bytes(_SH)
    monkeypatch.setattr(studio, "_installer_bundled_candidates", lambda n: [tmp_path / n])
    monkeypatch.setattr(studio, "_fetch_installer", lambda *a, **k: b"FETCHED")
    runs = _record_runs(monkeypatch, studio)
    studio._refresh_desktop_shortcuts()
    assert runs == [(["bash", "-s", "--", "--shortcuts-only"], b"FETCHED")]


def test_failed_fetch_falls_back_to_the_bundled_copy(monkeypatch, tmp_path):
    studio = _posix(monkeypatch, tmp_path)
    bundled = tmp_path / "install.sh"
    bundled.write_bytes(_SH)
    monkeypatch.setattr(studio, "_installer_bundled_candidates", lambda n: [tmp_path / n])
    monkeypatch.setattr(studio, "_fetch_installer", lambda *a, **k: None)
    runs = _record_runs(monkeypatch, studio)
    studio._refresh_desktop_shortcuts()
    assert runs == [(["bash", str(bundled), "--shortcuts-only"], None)]


def test_offline_with_no_bundled_copy_skips_instead_of_guessing(monkeypatch, tmp_path, capsys):
    studio = _posix(monkeypatch, tmp_path)
    monkeypatch.setattr(studio, "_installer_bundled_candidates", lambda n: [tmp_path / n])
    monkeypatch.setattr(studio, "_fetch_installer", lambda *a, **k: None)
    monkeypatch.setattr(
        studio.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("nothing to run")),
    )
    studio._refresh_desktop_shortcuts()
    assert "skipped" in capsys.readouterr().out


def test_opt_out_pins_the_refresh_to_the_bundled_installer(monkeypatch, tmp_path):
    studio = _posix(monkeypatch, tmp_path)
    monkeypatch.setenv("UNSLOTH_NO_REMOTE_INSTALLER", "1")
    monkeypatch.setattr(
        studio.urllib.request,
        "build_opener",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not open a connection")),
    )
    assert studio._fetch_installer("install.sh") is None


# ── what is allowed to reach a shell ───────────────────────────────────────────────


def test_non_installer_responses_are_not_executed():
    """A captive portal or error page must never be piped into bash."""
    studio = _studio()
    rejected = [
        None,
        b"",
        b"<!DOCTYPE html><html><body>Sign in to WiFi</body></html>" + b"x" * 4000,
        b"  <html>" + b"x" * 4000,
        b"rm -rf ~" + b"\n" * 4000,  # right size, wrong content
        _SH[:200],  # truncated
    ]
    for body in rejected:
        assert not studio._looks_like_installer(body, "install.sh"), body[:40] if body else body
    assert studio._looks_like_installer(_SH, "install.sh")


def test_oversized_and_bad_responses_return_none(monkeypatch, tmp_path):
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

    for body, why in (
        (b"x" * (studio._INSTALLER_MAX_BYTES + 1), "oversized"),
        (b"<html>not an installer</html>" + b"x" * 4000, "not an installer"),
    ):

        class _Opener:
            def open(self, *a, **k):
                return _Response(body)

        monkeypatch.setattr(studio.urllib.request, "build_opener", lambda *a, **k: _Opener())
        assert studio._fetch_installer("install.sh") is None, why
