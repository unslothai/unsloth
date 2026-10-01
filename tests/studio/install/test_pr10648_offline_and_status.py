# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PR #10648: a release lookup that could not answer must keep an intact install.

Two halves, because the feature is a contract between two programs.

PART 1 drives ``install_whisper_prebuilt.main()`` -- the same entrypoint setup.sh
runs -- with every network failure injected at the module's own primitives, and
asserts the exit code, the token in the log, and that the tree on disk did not
move. The wrap this PR added (``fetch_release_for_install``, ``_release_plan_for_host``)
is what turns a ``URLError`` / ``RuntimeError`` into the ``PrebuiltFallback`` the
keep path reads, so the assertions that matter most are the negative ones: with no
install, or a broken one, every single failure mode must still exit non-zero. An
offline update reporting success over nothing installed is the outcome this file
exists to rule out.

PART 2 executes the real ``setup.sh`` and ``setup.ps1`` status blocks, extracted
from the shipped scripts, and checks the label each exit-code/output pair produces
in BOTH shells. ``pwsh`` is present on Linux CI, so the Windows arm is measured
rather than asserted from source text.

No test here touches the network: every exit is stubbed, and each stub records the
URLs it was asked for so a test that intercepted nothing fails instead of passing.
"""

from __future__ import annotations

import email.message
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import urllib.error
from pathlib import Path

import pytest

from _pr10648_helpers import PACKAGE_ROOT, git, load_studio_module
from unsloth_pwsh_runner import run_pwsh


MODULE_PATH = PACKAGE_ROOT / "studio" / "install_whisper_prebuilt.py"

# A DISTINCT sys.modules name: test_install_whisper_prebuilt_logic.py owns
# "studio_install_whisper_prebuilt", and -n 4's per-worker cache would share the monkeypatches.
M = load_studio_module(
    "studio_install_whisper_prebuilt_pr10648_offline", "install_whisper_prebuilt.py"
)

HostInfo = M.HostInfo

# The merge base of studio-prebuilt-precheck, for the pre-PR comparison below.
MERGE_BASE = "191b69c12b4434b5247f1fd7a455b4a760b169ae"

RELEASE_TAG = "v1.9.1-unsloth.1"
UPSTREAM_TAG = "v1.9.1"
SOURCE_COMMIT = "0" * 40
STUDIO_PROTOCOL = "inference/multipart-v1"

# The two substrings the setup scripts grep. Spelled out here so a test that stops matching the
# installer is a test that stops matching the shipped scripts too.
KEEP_TOKEN = "keeping the existing complete install"
MATCH_TOKEN = "already matches"
FAIL_TOKEN = "prebuilt install failed"


def _host(
    whisper_os: str,
    whisper_arch: str,
    *,
    macos_version = None,
) -> HostInfo:
    return HostInfo(
        system = {"linux": "Linux", "macos": "Darwin", "windows": "Windows"}[whisper_os],
        machine = whisper_arch,
        whisper_os = whisper_os,
        whisper_arch = whisper_arch,
        archive_ext = ".zip" if whisper_os == "windows" else ".tar.gz",
        is_windows = whisper_os == "windows",
        is_macos = whisper_os == "macos",
        is_apple_silicon = whisper_os == "macos" and whisper_arch == "arm64",
        macos_version = macos_version,
    )


def _manifest(artifacts: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "component": "whisper.cpp",
        "studio_protocol": STUDIO_PROTOCOL,
        "upstream_tag": UPSTREAM_TAG,
        "source_commit": SOURCE_COMMIT,
        "artifacts": artifacts,
    }


def _artifact(host: HostInfo, asset: str, sha256: str, **extra) -> dict:
    artifact = {
        "os": host.whisper_os,
        "arch": host.whisper_arch,
        "backend": "cpu",
        "asset": asset,
        "sha256": sha256,
        "runtime_line": None,
    }
    artifact.update(extra)
    return artifact


def _build_cpu_bundle(tmp_path: Path, host: HostInfo) -> tuple[Path, str, str]:
    """A real archive on disk: the install path extracts and hashes it for real."""
    asset = M.whisper_asset_name(RELEASE_TAG, host, "cpu")
    archive = tmp_path / asset
    with tarfile.open(archive, "w:gz") as tar:
        for name, data, mode in (
            (M.server_binary_name(host), b"#!/bin/sh\necho whisper\n", 0o755),
            ("libwhisper.so", b"dummy-libwhisper", 0o644),
            ("libggml-base.so", b"dummy-libggml", 0o644),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = mode
            tar.addfile(info, io.BytesIO(data))
    return archive, asset, M.sha256_file(archive)


def _bundle(host: HostInfo, asset: str, sha256: str, **artifact_extra):
    manifest = M.parse_manifest(_manifest([_artifact(host, asset, sha256, **artifact_extra)]))
    return M.ReleaseBundle(
        repo = M.DEFAULT_PUBLISHED_REPO,
        release_tag = RELEASE_TAG,
        manifest = manifest,
        asset_urls = {asset: f"https://example.invalid/{asset}"},
    )


def _seed_install(tmp_path: Path, host: HostInfo, **artifact_extra) -> Path:
    """Install a real prebuilt tree + marker, with every patch undone afterwards.

    The seeding patches live in their own MonkeyPatch context so the failure injection the
    test then installs is the ONLY thing standing between the installer and the network.
    """
    archive, asset, sha256 = _build_cpu_bundle(tmp_path, host)
    install_dir = tmp_path / "whisper.cpp"
    bundle = _bundle(host, asset, sha256, **artifact_extra)
    checksums = {asset: sha256}

    def fake_download(url, destination):
        destination.parent.mkdir(parents = True, exist_ok = True)
        destination.write_bytes(archive.read_bytes())

    with pytest.MonkeyPatch.context() as seeding:
        seeding.delenv("UNSLOTH_PREBUILT_FULL_CHECK", raising = False)
        seeding.setattr(M, "detect_host", lambda: host)
        seeding.setattr(M, "download_file", fake_download)
        seeding.setattr(
            M,
            "fetch_release_for_install",
            lambda repo, *, published_release_tag = None: (bundle, checksums),
        )
        assert M.install_prebuilt(install_dir, backend = "cpu") == M.EXIT_SUCCESS
    assert M.metadata_path(install_dir).is_file()
    return install_dir


def _http_error(code: int, url: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(url, code, f"injected {code}", email.message.Message(), None)


# Every failure mode, named once. The value is what the injected primitives do.
FAILURE_MODES = (
    "offline",  # urllib.error.URLError: no network at all
    "timeout",  # socket.timeout on both the CDN HEAD and every GET
    "http_403",  # api.github.com rate limit -> fetch_json raises RuntimeError
    "http_429",  # api.github.com secondary rate limit -> same RuntimeError
    "http_404",  # the repo/release does not answer
    "http_500",  # GitHub 5xx
    "truncated_json",  # a body that stops mid-object
    "malformed_json",  # a captive-portal/proxy HTML page where JSON was expected
    "empty_releases",  # a well-formed answer listing no releases at all
)


class _InjectedNetwork:
    """Every network exit this installer has, replaced by one injected failure.

    There are exactly two primitives, and both are resolved through ``_OPS`` against the
    installer module's own globals, so setting them here is the real seam rather than an
    unused alias:

      * ``download_bytes`` -- every JSON GET, api.github.com and the release CDN alike
        (``core.fetch_json`` and ``core.fetch_download_host_json`` both route through it).
      * ``_URL_OPENER`` -- the bare HEAD ``core.download_host_latest_release_tag`` uses to
        read /releases/latest without spending API quota.

    ``calls`` is the proof the injection fired. A test whose patch intercepted nothing would
    otherwise be green while the real code took some other path.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls: list[tuple[str, str]] = []
        self.downloads: list[str] = []

    def _raise_transport(self, url: str) -> None:
        if self.mode == "offline":
            raise urllib.error.URLError(OSError("injected: network is unreachable"))
        if self.mode == "timeout":
            raise socket.timeout("injected: timed out")

    # ── the /releases/latest HEAD ──
    def open(
        self,
        request,
        timeout = None,
    ):
        url = getattr(request, "full_url", str(request))
        self.calls.append(("head", url))
        self._raise_transport(url)
        # Every remaining mode is about api.github.com, so the CDN shortcut declines first. A 404
        # is how download_host_latest_release_tag says "cannot name the release": None, not a raise.
        raise _http_error(404, url)

    # ── every JSON GET ──
    def download_bytes(self, url, **_kwargs) -> bytes:
        self.calls.append(("get", url))
        self._raise_transport(url)
        if self.mode.startswith("http_"):
            raise _http_error(int(self.mode.split("_")[1]), url)
        if self.mode == "truncated_json":
            return b'{"tag_name": "v1.9.1-unslo'
        if self.mode == "malformed_json":
            return b"<html><head><title>502 Bad Gateway</title></head></html>"
        if self.mode == "empty_releases":
            return b"[]"
        raise AssertionError(f"unhandled injection mode {self.mode!r}")

    # ── the pre-check's own HEAD, which lives on the llama module ──
    def llama_latest_tag(self, repo: str) -> str | None:
        self.calls.append(("precheck", repo))
        self._raise_transport(repo)
        raise _http_error(404, f"https://github.com/{repo}/releases/latest")

    def download_file(self, url, destination):  # pragma: no cover - asserted never to run
        self.downloads.append(str(url))
        raise AssertionError(f"a keep/fail path must not download anything, but fetched {url}")

    @property
    def install_calls(self) -> list[tuple[str, str]]:
        """Calls made by the INSTALL path, i.e. not the marker-only pre-check.

        A fully offline run never reaches ``download_bytes`` at all -- the CDN HEAD that
        resolves /releases/latest raises first -- so this, not a GET count, is what proves
        the injection fired where the PR's wrap had to catch it."""
        return [(kind, url) for kind, url in self.calls if kind != "precheck"]


def _inject(monkeypatch, mode: str) -> _InjectedNetwork:
    """Close every network exit, then prove the one that mattered was used.

    Patching only ``M.download_bytes`` is the trap this helper exists to avoid: the
    installer does ``fetch_json = llama.fetch_json`` and ``download_bytes =
    llama.download_bytes`` at import, so ``core.fetch_json`` resolves through the LLAMA
    module's globals and an injection on the whisper alias intercepts nothing -- the first
    version of this file reached api.github.com for real and still looked green until
    ``calls`` was read back and showed a live release tag in the URL.

    So: both modules' primitives, plus ``prebuilt_core._URL_OPENER`` underneath them, which
    every HTTP call in all three modules ultimately goes through.
    """
    net = _InjectedNetwork(mode)
    for module in (M, M.llama):
        monkeypatch.setattr(module, "download_bytes", net.download_bytes)
        monkeypatch.setattr(module, "download_file", net.download_file)
        monkeypatch.setattr(module, "_URL_OPENER", net, raising = False)
        # One attempt: the JSON decode path otherwise retries with real sleeps.
        monkeypatch.setattr(module, "JSON_FETCH_ATTEMPTS", 1, raising = False)
        monkeypatch.setattr(module, "HTTP_FETCH_ATTEMPTS", 1, raising = False)
    monkeypatch.setattr(M.core, "_URL_OPENER", net)
    monkeypatch.setattr(M.llama, "_download_host_latest_release_tag", net.llama_latest_tag)
    monkeypatch.delenv("UNSLOTH_PREBUILT_FULL_CHECK", raising = False)
    monkeypatch.delenv("UNSLOTH_WHISPER_FORCE_COMPILE", raising = False)
    return net


def _run_cli(monkeypatch, capsys, host: HostInfo, install_dir: Path, *extra: str):
    """Drive main(), the entrypoint setup.sh runs, and return (exit code, log text)."""
    monkeypatch.setattr(M, "detect_host", lambda: host)
    # main() flips this module global; register the current value so it is restored.
    monkeypatch.setattr(M, "_LOG_TO_STDOUT", M._LOG_TO_STDOUT)
    capsys.readouterr()
    code = M.main(["--install-dir", str(install_dir), "--backend", "cpu", *extra])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def _tree_snapshot(install_dir: Path) -> dict[str, bytes]:
    if not install_dir.exists():
        return {}
    return {
        str(path.relative_to(install_dir)): path.read_bytes()
        for path in sorted(install_dir.rglob("*"))
        if path.is_file()
    }


def _partial_artifacts(install_dir: Path) -> list[str]:
    """Anything a half-finished install would leave: staging trees and atomic temps."""
    if not install_dir.exists():
        return []
    return sorted(
        str(path.relative_to(install_dir))
        for path in install_dir.rglob("*")
        if ".tmp-" in path.name or path.name.startswith(".tmp-")
    )


# ══ Part 1: network failure modes ════════════════════════════════════════════════════


@pytest.mark.parametrize("mode", FAILURE_MODES)
def test_every_lookup_failure_keeps_an_intact_install(tmp_path, monkeypatch, capsys, mode):
    """An unpinned update whose lookup could not answer keeps the tree and exits 0.

    This is the whole point of the PR: before it, ``URLError`` and ``fetch_json``'s
    ``RuntimeError`` escaped ``install_prebuilt`` uncaught and setup printed
    "prebuilt install failed" over a healthy install.

    Note which modes are in this list. 404 and 500 are here too, because with no release
    pin they are also "the lookup could not answer": nothing named a release, so nothing
    established that the install on disk is stale. The pinned case below is where a 404
    stops being an unavailability and starts being an answer.
    """
    host = _host("linux", "x64")
    install_dir = _seed_install(tmp_path, host)
    before = _tree_snapshot(install_dir)
    marker_before = M.metadata_path(install_dir).read_bytes()

    net = _inject(monkeypatch, mode)
    code, log = _run_cli(monkeypatch, capsys, host, install_dir)

    assert net.install_calls, f"{mode}: the injection never fired; the test intercepted nothing"
    assert code == M.EXIT_SUCCESS, f"{mode}: exit {code}\n{log}"
    # The exact substring studio/setup.sh:3636 and studio/setup.ps1:6381 grep for.
    assert KEEP_TOKEN in log, f"{mode}: {log}"
    assert FAIL_TOKEN not in log, f"{mode}: {log}"
    # Not "already matches": that arm names a release this run never fetched.
    assert MATCH_TOKEN not in log, f"{mode}: {log}"

    assert net.downloads == [], f"{mode}: a kept install downloaded {net.downloads}"
    assert _tree_snapshot(install_dir) == before, f"{mode}: the kept tree moved"
    assert M.metadata_path(install_dir).read_bytes() == marker_before, f"{mode}: marker rewritten"
    assert _partial_artifacts(install_dir) == [], f"{mode}: half-installed leftovers"


@pytest.mark.parametrize("mode", FAILURE_MODES)
@pytest.mark.parametrize(
    "damage",
    ["absent", "no_marker", "zero_byte_server", "tampered_marker", "server_deleted"],
)
def test_no_failure_mode_reports_success_without_a_working_install(
    tmp_path, monkeypatch, capsys, mode, damage
):
    """The worst possible outcome, ruled out for every mode x every damage shape.

    An update that cannot reach the network and reports success while nothing usable is
    installed would leave setup.sh printing "prebuilt installed" over an empty directory
    and dictation silently broken. Every one of these must exit non-zero and say
    "prebuilt install failed" -- the token setup.sh's else-arm reports.
    """
    host = _host("linux", "x64")
    if damage == "absent":
        install_dir = tmp_path / "whisper.cpp"
    else:
        install_dir = _seed_install(tmp_path, host)
        marker = M.metadata_path(install_dir)
        server = M.installed_server_path(install_dir, host)
        if damage == "no_marker":
            marker.unlink()
        elif damage == "zero_byte_server":
            # Keeps its mode bits, so neither the execute-bit nor the existence check sees it.
            server.write_bytes(b"")
        elif damage == "server_deleted":
            server.unlink()
        elif damage == "tampered_marker":
            payload = json.loads(marker.read_text(encoding = "utf-8"))
            payload["release_tag"] = "v9.9.9-unsloth.99"
            marker.write_text(json.dumps(payload), encoding = "utf-8")

    net = _inject(monkeypatch, mode)
    code, log = _run_cli(monkeypatch, capsys, host, install_dir)

    assert net.install_calls, f"{mode}/{damage}: the injection never fired"
    assert (
        code != M.EXIT_SUCCESS
    ), f"{mode}/{damage}: reported success with no working install\n{log}"
    assert code == M.EXIT_ERROR, f"{mode}/{damage}: exit {code}\n{log}"
    assert KEEP_TOKEN not in log, f"{mode}/{damage}: claimed to keep a broken install\n{log}"
    assert MATCH_TOKEN not in log, f"{mode}/{damage}: {log}"
    assert FAIL_TOKEN in log, f"{mode}/{damage}: {log}"
    assert net.downloads == [], f"{mode}/{damage}: downloaded {net.downloads}"
    assert _partial_artifacts(install_dir) == [], f"{mode}/{damage}: half-installed leftovers"
    if damage == "absent":
        assert (
            not install_dir.exists() or _tree_snapshot(install_dir) == {}
        ), f"{mode}: a failed install left a tree behind"


@pytest.mark.parametrize("mode", FAILURE_MODES)
def test_an_explicitly_pinned_release_is_never_silently_kept(tmp_path, monkeypatch, capsys, mode):
    """A run that NAMED a release must not be answered with a different one.

    Keeping ignores the pin, so the pin has to fail instead. This is also the honest
    answer to the 404 question: a 404 on ``--published-release-tag v9.9.9-unsloth.99``
    is a deleted or misspelled tag -- an answer -- and it exits 1 rather than reporting
    the installed release as a success.
    """
    host = _host("linux", "x64")
    install_dir = _seed_install(tmp_path, host)
    before = _tree_snapshot(install_dir)

    net = _inject(monkeypatch, mode)
    code, log = _run_cli(
        monkeypatch, capsys, host, install_dir, "--published-release-tag", "v9.9.9-unsloth.99"
    )

    assert net.calls, f"{mode}: the injection never fired"
    assert code == M.EXIT_ERROR, f"{mode}: exit {code}\n{log}"
    assert KEEP_TOKEN not in log, f"{mode}: a pinned release was papered over\n{log}"
    assert FAIL_TOKEN in log, f"{mode}: {log}"
    # The intact tree is still not damaged -- refusing to keep is not the same as deleting.
    assert _tree_snapshot(install_dir) == before, f"{mode}: a failed pinned run moved the tree"
    assert _partial_artifacts(install_dir) == [], f"{mode}: half-installed leftovers"


@pytest.mark.parametrize(
    "extra_argv, env",
    [
        (("--force",), {}),
        (("--whisper-tag", "v1.9.7"), {}),
        ((), {"UNSLOTH_WHISPER_FORCE_COMPILE": "1"}),
    ],
    ids = ["force", "upstream-tag-pin", "force-compile"],
)
def test_an_explicit_request_never_keeps(tmp_path, monkeypatch, capsys, extra_argv, env):
    """Everything this RUN asked for that keeping would ignore fails instead.

    ``--force`` asked for a reinstall, ``--whisper-tag`` asked for a specific upstream
    version, and ``UNSLOTH_WHISPER_FORCE_COMPILE=1`` asked setup.sh to try a source build,
    which it only does after a non-zero exit. Reporting 0 would swallow all three.
    """
    host = _host("linux", "x64")
    install_dir = _seed_install(tmp_path, host)

    net = _inject(monkeypatch, "offline")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    code, log = _run_cli(monkeypatch, capsys, host, install_dir, *extra_argv)

    assert net.calls, "the injection never fired"
    assert code == M.EXIT_ERROR, f"exit {code}\n{log}"
    assert KEEP_TOKEN not in log, log
    assert FAIL_TOKEN in log, log


def test_the_keep_arm_names_the_reason_and_the_release_it_kept(tmp_path, monkeypatch, capsys):
    """The kept line has to be readable on its own: setup.sh prints one status line and
    throws the installer log away, so the release and the reason both belong in it."""
    host = _host("linux", "x64")
    install_dir = _seed_install(tmp_path, host)
    _inject(monkeypatch, "http_403")
    code, log = _run_cli(monkeypatch, capsys, host, install_dir)

    assert code == M.EXIT_SUCCESS, log
    assert KEEP_TOKEN in log
    assert RELEASE_TAG in log, log
    assert "update unavailable, existing prebuilt kept" in log, log
    assert "prebuilt update reason:" in log, log
    # The 403 arrives as fetch_json's RuntimeError, not an OSError; the PR's wrap is the only
    # thing that turns it into the PrebuiltFallback the keep path reads.
    assert "403" in log, log


def test_a_release_compatibility_error_is_never_papered_over(tmp_path, monkeypatch, capsys):
    """Exit 2 is a lookup that ANSWERED: no published bundle pairs with this runtime.

    setup.sh names both tags from exit 2, so converting it into a kept success would hide
    real release skew behind a warning about the network.
    """
    host = _host("linux", "x64")
    install_dir = _seed_install(tmp_path, host)
    _inject(monkeypatch, "offline")

    def incompatible(*_args, **_kwargs):
        raise M.ReleaseCompatibilityError("slim bundle requires llama.cpp b9999; installed b1000")

    monkeypatch.setattr(M, "_release_plan_for_host", incompatible)
    code, log = _run_cli(monkeypatch, capsys, host, install_dir)

    assert code == M.EXIT_INCOMPATIBLE, f"exit {code}\n{log}"
    assert KEEP_TOKEN not in log, log


def _macos_listing_failure(tmp_path, monkeypatch, host):
    """Newest published release incompatible with this Mac, release LISTING unavailable.

    ``_release_plan_for_host`` records the newest release's incompatibility in
    ``first_error``, then walks older releases to look for one this Mac can run. When the
    listing that drives the walk cannot be fetched, line 1424-1427 raises a new
    PrebuiltFallback and ``first_error`` is discarded.
    """
    newer_dir = tmp_path / "newer"
    newer_dir.mkdir(parents = True, exist_ok = True)
    archive, asset, sha256 = _build_cpu_bundle(newer_dir, host)
    # A newest release whose only artifact needs a macOS this host does not have.
    newest = _bundle(host, asset, sha256, min_os = "26.0")
    # Control: the SAME bundle is selectable on a new enough Mac, so the rejection below is the
    # min_os floor, not a manifest built wrong. Without it the scenario passes exercising nothing.
    M.select_artifact_with_cpu_fallback(
        newest.manifest, _host("macos", host.whisper_arch, macos_version = (26, 0)), "cpu"
    )
    with pytest.raises(M.PrebuiltFallback):
        M.select_artifact_with_cpu_fallback(newest.manifest, host, "cpu")
    monkeypatch.setattr(
        M,
        "fetch_release_for_install",
        lambda repo, *, published_release_tag = None: (newest, {asset: sha256}),
    )
    return _inject(monkeypatch, "offline")


def test_macos_incompatible_newest_plus_unreachable_listing_keeps_a_runnable_install(
    tmp_path, monkeypatch, capsys
):
    """The open question, settled: this is an IMPROVEMENT, not a false success.

    ``first_error`` describes the NEWEST release only. The walk-back loop exists precisely
    because an older release may still run here, so with the listing unreachable the
    installer has NOT established "your macOS is too old for any available build" -- it has
    established that it cannot tell. Discarding first_error for "could not list releases"
    is therefore the accurate verdict, not a downgrade of a real one.

    And the install that is kept is known to run on this host: ``_existing_install_is_intact``
    re-checks the marker's ``min_os`` against ``host.macos_version`` before keeping anything
    (test_macos_keep_refuses_an_install_below_this_hosts_floor below is the negative case).
    """
    host = _host("macos", "arm64", macos_version = (13, 0))
    install_dir = _seed_install(tmp_path, host, min_os = "13.0")
    before = _tree_snapshot(install_dir)

    net = _macos_listing_failure(tmp_path, monkeypatch, host)
    code, log = _run_cli(monkeypatch, capsys, host, install_dir)

    # The listing endpoint _published_release_tags walks, and nothing else: this is the call
    # whose failure discards first_error.
    assert [url for _kind, url in net.install_calls] == [
        "https://api.github.com/repos/unslothai/whisper.cpp/releases?per_page=100"
    ], net.install_calls
    assert code == M.EXIT_SUCCESS, f"exit {code}\n{log}"
    assert KEEP_TOKEN in log, log
    # The reason names the unavailability, so nobody reads it as a compatibility verdict.
    assert "could not list" in log, log
    # And the discarded first_error is NOT what the user is told; a kept install is.
    assert "no whisper.cpp prebuilt asset" not in log, log
    assert _tree_snapshot(install_dir) == before


def test_macos_incompatible_newest_plus_unreachable_listing_fails_without_an_install(
    tmp_path, monkeypatch, capsys
):
    """The other half of the same verdict: nothing on disk, nothing invented.

    If the discarded first_error had been converted into a false success, THIS is where it
    would show -- a Mac with no whisper install reporting exit 0 after a lookup that never
    found a compatible bundle. It exits 1.
    """
    host = _host("macos", "arm64", macos_version = (13, 0))
    install_dir = tmp_path / "whisper.cpp"

    net = _macos_listing_failure(tmp_path, monkeypatch, host)
    code, log = _run_cli(monkeypatch, capsys, host, install_dir)

    assert net.install_calls
    assert code == M.EXIT_ERROR, f"exit {code}\n{log}"
    assert KEEP_TOKEN not in log, log
    assert FAIL_TOKEN in log, log


def test_macos_keep_refuses_an_install_below_this_hosts_floor(tmp_path, monkeypatch, capsys):
    """A tree carried onto an older Mac is not "intact" and must not be kept offline.

    Without this, the keep path would hold an install that cannot load, which is the shape
    a genuine false success would take.
    """
    build_host = _host("macos", "arm64", macos_version = (15, 0))
    install_dir = _seed_install(tmp_path, build_host, min_os = "15.0")
    older_mac = _host("macos", "arm64", macos_version = (13, 0))

    # Only the floor changed: the same tree is intact on the Mac it was installed on. Asserting the
    # predicate keeps the exit-1 below from passing for an unrelated reason.
    intact = dict(
        published_repo = M.DEFAULT_PUBLISHED_REPO,
        requested_backend = "cpu",
    )
    assert M._existing_install_is_intact(install_dir, build_host, **intact) is not None
    assert M._existing_install_is_intact(install_dir, older_mac, **intact) is None

    net = _inject(monkeypatch, "offline")
    code, log = _run_cli(monkeypatch, capsys, older_mac, install_dir)

    assert net.calls
    assert code == M.EXIT_ERROR, f"exit {code}\n{log}"
    assert KEEP_TOKEN not in log, log


def _git(*args: str) -> subprocess.CompletedProcess:
    return git(*args, text = True, timeout = 60)


requires_merge_base = pytest.mark.skipif(
    shutil.which("git") is None or _git("cat-file", "-e", f"{MERGE_BASE}^{{commit}}").returncode,
    reason = f"the PR merge base {MERGE_BASE} is not in this checkout",
)


@requires_merge_base
def test_pre_pr_an_unreachable_release_listing_escaped_as_an_uncaught_oserror():
    """Evidence for the verdict above, from the code this PR replaced.

    Pre-PR, ``_release_plan_for_host`` iterated ``_published_release_tags(...)`` directly
    and ``install_prebuilt`` had no except arm at all, so an unreachable listing left the
    OSError to main()'s catch-all: "unexpected error", exit 1, over an intact install --
    which setup.sh then reported as "prebuilt install failed".

    So the macOS case is not a real verdict being converted into a false success. Pre-PR
    there was no verdict to lose: first_error was discarded then too (the OSError simply
    replaced it by propagating), and the user got a failure instead of a working runtime.
    """
    pre = _git("show", f"{MERGE_BASE}:studio/install_whisper_prebuilt.py")
    assert pre.returncode == 0, pre.stderr
    source = pre.stdout

    # The loop was unguarded: no try/except between the planner and the listing fetch.
    assert (
        "    for release_tag in _published_release_tags(published_repo):" in source
    ), "the pre-PR planner did not iterate the listing directly; re-read the comparison"
    assert "could not list" not in source, "pre-PR already had the listing guard"
    # install_prebuilt had no keep arm, so nothing downgraded a lookup failure to exit 0.
    pre_install = source[source.index("def install_prebuilt(") :]
    pre_install = pre_install[: pre_install.index("\ndef ")]
    assert "except PrebuiltFallback" not in pre_install, pre_install
    assert KEEP_TOKEN not in source, "pre-PR already logged the keep token"

    # And post-PR it is guarded, in the same function.
    post = MODULE_PATH.read_text(encoding = "utf-8")
    assert "except (OSError, RuntimeError) as exc:" in post
    guard = 'raise PrebuiltFallback(f"could not list {published_repo} releases: {exc}") from exc'
    assert guard in post


def test_the_keep_arm_sits_beside_the_status_arms_it_did_not_replace():
    """The pre-existing tokens must still mean exactly what they meant before.

    ``already matches`` and llama's two arms carry other behaviour (node greps the same
    token at setup.sh:1261), so this PR's keep arm is only safe next to them rather than
    written over them.

    Stated on the shipped text, not on a diff. The original spelling of this test read
    ``git diff <the PR's merge base> -- studio/setup.*`` and asserted that no line carrying
    a status decision appeared as a REMOVAL, which is a property of one branch and not of
    the product: on ``main`` that diff is everything the installers have done since
    2026-09-09, so it went red the moment the PR landed on a main that had moved, and it
    said so about whichever unrelated commit happened to touch a ``-match`` line. CI never
    reported it because ``actions/checkout`` is shallow there and the merge base is absent,
    so ``requires_merge_base`` skipped the whole thing. What the assertions below keep is
    the part that is checkable forever: each arm, each guard, and exactly one keep arm per
    runtime per script -- a rewritten or duplicated arm still fails here.
    """
    setup_sh = (PACKAGE_ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
    setup_ps1 = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")

    # The keep arm this PR added: one per runtime per script, and guarded with each shell's
    # own test rather than, say, both scripts growing the sh spelling.
    sh_keeps = [line for line in setup_sh.splitlines() if KEEP_TOKEN in line]
    ps1_keeps = [line for line in setup_ps1.splitlines() if KEEP_TOKEN in line]
    assert len(sh_keeps) == 2, sh_keeps
    assert len(ps1_keeps) == 2, ps1_keeps
    assert all("grep -Fq" in line for line in sh_keeps), sh_keeps
    assert all("-match" in line for line in ps1_keeps), ps1_keeps

    # The llama arms and node's grep are still exactly where they were.
    assert 'grep -Fq "already matches" "$_PREBUILT_LOG"' in setup_sh
    assert 'grep -Fq "already matches" "$_NODE_LOG"' in setup_sh
    assert f'$prebuiltOutput -match "{MATCH_TOKEN}"' in setup_ps1
    for script in (setup_sh, setup_ps1):
        assert (
            script.count("update unavailable, existing prebuilt kept") == 2
        ), "llama and whisper each have exactly one keep arm"


# ══ Part 2: the status contract, executed in both shells ═════════════════════════════

_SH_WHISPER_START = 'if [ "$_WHISPER_STATUS" -eq 0 ]; then'
_PS1_WHISPER_START = "if ($whisperExit -eq 0) {"
_SH_LLAMA_START = 'if [ "$_PREBUILT_STATUS" -eq 0 ]; then'
_PS1_LLAMA_START = "if ($prebuiltExit -eq 0) {"

# Stand-ins for the setup helpers the blocks call. `step` echoes both arguments, which is what
# distinguishes "llama reported the keep" from "whisper reported it".
_SH_HARNESS = """
set -u
C_OK=""; C_WARN=""; C_ERR=""
_NEED_LLAMA_SOURCE_BUILD=false
_LLAMA_CPP_NO_SPACE=false
_LLAMA_CPP_DEGRADED=false
_explicit_llama_backend=""
_STUDIO_HOME_IS_CUSTOM=false
# The flag the runtime children switched to when UNSLOTH_HOME arrived: the guards inside the
# extracted block read it, and setup.sh derives it in the section this harness stands in for.
_RUNTIME_ROOT_IS_CUSTOM=false
_STUDIO_OWNED_MARKER=".unsloth-owned"
_WHISPER_RECOVERED=false
step() { echo "step|$1|$2"; }
substep() { echo "substep|$1"; }
verbose_substep() { :; }
run_quiet_no_exit() { return 1; }
print_llama_error_log() { :; }
print_installed_llama_prebuilt_release() { :; }
_has_local_llama_server() { return 1; }
setup_fail() { echo "setup_fail|$1"; exit "$1"; }
"""

_PS1_HARNESS = """
$ErrorActionPreference = "Stop"
$NeedLlamaSourceBuild = $false
$script:LlamaCppDegraded = $false
$StudioHomeIsCustom = $false
# Same reason as _RUNTIME_ROOT_IS_CUSTOM in the shell harness above.
$RuntimeRootIsCustom = $false
function step { param($a, $b, $c) Write-Output "step|$a|$b" }
function substep { param($a, $b) Write-Output "substep|$a" }
function Write-LlamaFailureLog { param($Output) }
function Mark-StudioOwned { param($Path) }
function Get-InstalledLlamaPrebuiltRelease { param($InstallDir) return $null }
function Test-PathQuiet { param($p, $t) return $false }
function Exit-SetupFailure {
    param($Message, $Code = 1)
    Write-Output "setup_fail|$Code"
    exit $Code
}
"""


def _extract_block(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start : end + len(end_marker)]


def _sh_block(name: str) -> str:
    setup_sh = (PACKAGE_ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
    start = setup_sh.index(_SH_WHISPER_START if name == "whisper" else _SH_LLAMA_START)
    # Both chains end at the first dedented "    fi"; every nested fi inside is indented deeper.
    end = setup_sh.index("\n    fi\n", start)
    return setup_sh[start : end + len("\n    fi\n")]


def _ps1_block(name: str) -> str:
    setup_ps1 = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")
    if name == "whisper":
        # The exit-2 arm ends with the same words followed by "} else {", so the trailing
        # newline is what pins this to the final closing brace of the chain.
        return _extract_block(setup_ps1, _PS1_WHISPER_START, 'remain available" "Yellow"\n    }\n')
    return _extract_block(setup_ps1, _PS1_LLAMA_START, 'retry setup."\n        }')


def _run_bash(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], capture_output = True, text = True, timeout = 60)


def _steps(stdout: str) -> dict[str, str]:
    """The component -> label decision each block reached."""
    found = {}
    for line in stdout.splitlines():
        if line.startswith("step|"):
            _, component, label = line.split("|", 2)
            found[component] = label
    return found


def _sh_whisper_script(tmp_path: Path, status: int, output: str) -> str:
    log_path = tmp_path / "whisper.log"
    log_path.write_text(output, encoding = "utf-8")
    return "\n".join(
        [
            _SH_HARNESS,
            f'SCRIPT_DIR="{tmp_path}"',
            f'UNSLOTH_HOME="{tmp_path}"',
            f'WHISPER_CPP_DIR="{tmp_path / "whisper.cpp"}"',
            f'_WHISPER_LOG="{log_path}"',
            f"_WHISPER_STATUS={status}",
            _sh_block("whisper"),
        ]
    )


def _ps1_whisper_script(tmp_path: Path, status: int, output: str) -> str:
    return "\n".join(
        [
            _PS1_HARNESS,
            f'$WhisperCppDir = "{tmp_path / "whisper.cpp"}"',
            f'$LlamaCppDir = "{tmp_path / "llama.cpp"}"',
            f"$whisperExit = {status}",
            "$whisperOutput = @'",
            output,
            "'@",
            _ps1_block("whisper"),
        ]
    )


def _run_ps1(
    tmp_path: Path,
    script: str,
    name: str = "status.ps1",
):
    script_path = tmp_path / name
    script_path.write_text(script, encoding = "utf-8")
    # run_pwsh, not subprocess.run: a pwsh killed at startup returns rc -6 with empty stdout,
    # which this file would otherwise report as setup.ps1 choosing the wrong label.
    return run_pwsh(
        [
            shutil.which("pwsh") or "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(script_path),
        ],
        capture_output = True,
        text = True,
        timeout = 120,
    )


requires_pwsh = pytest.mark.skipif(
    shutil.which("pwsh") is None,
    reason = "pwsh is required to execute the setup.ps1 status block",
)
requires_bash = pytest.mark.skipif(
    shutil.which("bash") is None or os.name == "nt",
    reason = "setup.sh is the POSIX installer",
)

# One table, run in both shells: `output` is what the installer printed, `label` what the user
# must be told. The last three rows are the traps, where the two disagree.
STATUS_CASES = [
    (
        "up-to-date",
        0,
        "[whisper-prebuilt] existing whisper.cpp install already matches v1.9.1-unsloth.1 (cpu)",
        "prebuilt up to date",
    ),
    (
        "kept",
        0,
        "[whisper-prebuilt] whisper.cpp update unavailable, existing prebuilt kept; keeping the "
        "existing complete install of v1.9.1-unsloth.1",
        "update unavailable, existing prebuilt kept",
    ),
    (
        "installed",
        0,
        "[whisper-prebuilt] installed whisper.cpp v1.9.2-unsloth.3 (cpu)",
        "prebuilt installed",
    ),
    (
        "kept-wins-over-installed-text",
        0,
        "[whisper-prebuilt] downloading whisper-v1.9.1-unsloth.1-linux-x64-cpu.tar.gz\n"
        "[whisper-prebuilt] whisper.cpp update unavailable, existing prebuilt kept; keeping the "
        "existing complete install of v1.9.1-unsloth.1",
        "update unavailable, existing prebuilt kept",
    ),
]

# Non-zero exits: the label must never claim success, whatever the output says.
FAILURE_CASES = [
    ("plain-failure", 1, "[whisper-prebuilt] prebuilt install failed: could not fetch release"),
    (
        "failure-carrying-the-match-token",
        1,
        "[whisper-prebuilt] checking whether the install already matches\n"
        "[whisper-prebuilt] prebuilt install failed: checksum mismatch",
    ),
    (
        "failure-carrying-the-keep-token",
        1,
        "[whisper-prebuilt] considered keeping the existing complete install\n"
        "[whisper-prebuilt] prebuilt install failed: archive member escaped destination",
    ),
    (
        "unexpected-exit-carrying-both",
        137,
        "[whisper-prebuilt] already matches\n"
        "[whisper-prebuilt] keeping the existing complete install",
    ),
]

_FAILURE_LABEL_PREFIX = "prebuilt install failed"


@requires_bash
@pytest.mark.parametrize(
    "name, status, output, label", STATUS_CASES, ids = [case[0] for case in STATUS_CASES]
)
def test_setup_sh_status_labels(tmp_path, name, status, output, label):
    result = _run_bash(_sh_whisper_script(tmp_path, status, output))
    assert result.returncode == 0, result
    assert _steps(result.stdout).get("whisper.cpp") == label, result.stdout


@requires_pwsh
@pytest.mark.parametrize(
    "name, status, output, label", STATUS_CASES, ids = [case[0] for case in STATUS_CASES]
)
def test_setup_ps1_status_labels(tmp_path, name, status, output, label):
    result = _run_ps1(tmp_path, _ps1_whisper_script(tmp_path, status, output))
    assert result.returncode == 0, result
    assert _steps(result.stdout).get("whisper.cpp") == label, (result.stdout, result.stderr)


@requires_bash
@pytest.mark.parametrize(
    "name, status, output", FAILURE_CASES, ids = [case[0] for case in FAILURE_CASES]
)
def test_setup_sh_never_reports_success_for_a_non_zero_exit(tmp_path, name, status, output):
    """Success TEXT on a failed run must not become a success LABEL.

    The installer writes its progress log and its error to the same stream, so a run that
    got as far as printing either token and then died is exactly the shape that would turn
    a broken dictation runtime into a green line if the grep ran before the exit check.
    """
    result = _run_bash(_sh_whisper_script(tmp_path, status, output))
    assert result.returncode == 0, result
    label = _steps(result.stdout).get("whisper.cpp")
    assert label is not None, result.stdout
    assert label.startswith(_FAILURE_LABEL_PREFIX), label
    assert "up to date" not in label, label
    assert "existing prebuilt kept" not in label, label


@requires_pwsh
@pytest.mark.parametrize(
    "name, status, output", FAILURE_CASES, ids = [case[0] for case in FAILURE_CASES]
)
def test_setup_ps1_never_reports_success_for_a_non_zero_exit(tmp_path, name, status, output):
    result = _run_ps1(tmp_path, _ps1_whisper_script(tmp_path, status, output))
    assert result.returncode == 0, (result.stdout, result.stderr)
    label = _steps(result.stdout).get("whisper.cpp")
    assert label is not None, (result.stdout, result.stderr)
    assert label.startswith(_FAILURE_LABEL_PREFIX), label
    assert "up to date" not in label, label
    assert "existing prebuilt kept" not in label, label


@requires_bash
@requires_pwsh
@pytest.mark.parametrize(
    "name, status, output, label", STATUS_CASES, ids = [case[0] for case in STATUS_CASES]
)
def test_both_shells_agree_on_the_label(tmp_path, name, status, output, label):
    """Windows and POSIX must tell the user the same thing.

    Asserting each side against a constant leaves them free to drift apart via the
    constant; comparing the two measured labels is what makes "mirrored" a measurement.
    """
    sh = _run_bash(_sh_whisper_script(tmp_path, status, output))
    ps = _run_ps1(tmp_path, _ps1_whisper_script(tmp_path, status, output))
    assert sh.returncode == 0 and ps.returncode == 0, (sh, ps.stdout, ps.stderr)
    assert _steps(sh.stdout).get("whisper.cpp") == _steps(ps.stdout).get("whisper.cpp") == label


@requires_bash
@requires_pwsh
def test_a_success_token_inside_an_unrelated_line_still_decides_the_label(tmp_path):
    """Characterisation, and a real limit of the contract, in both shells.

    The grep is a plain substring on the WHOLE log, so any line that happens to contain
    "already matches" wins the first arm even when the run actually installed something.
    No shipped installer line does this today (the only producers are
    existing_install_current_without_plan and the keep arm), so it is a latent precision
    limit rather than a live bug -- and it can only ever choose between two SUCCESS labels,
    never turn a failure into one, which the FAILURE_CASES above pin down.

    This is here so that a future log line containing the token fails loudly instead of
    silently relabelling a fresh install as "up to date".
    """
    output = (
        "[whisper-prebuilt] no published release already matches this host's runtime; "
        "downloading\n[whisper-prebuilt] installed whisper.cpp v1.9.2-unsloth.3 (cpu)"
    )
    sh = _run_bash(_sh_whisper_script(tmp_path, 0, output))
    ps = _run_ps1(tmp_path, _ps1_whisper_script(tmp_path, 0, output))
    assert _steps(sh.stdout).get("whisper.cpp") == "prebuilt up to date"
    assert _steps(ps.stdout).get("whisper.cpp") == "prebuilt up to date"


@requires_bash
@requires_pwsh
def test_the_two_shells_differ_on_token_case(tmp_path):
    """Characterisation of a real divergence with no live trigger.

    ``grep -Fq`` is case sensitive; PowerShell's ``-match`` is case INSENSITIVE by default.
    So a log saying "Already Matches" is reported as a fresh install on Linux and as up to
    date on Windows. Every line the installer actually emits is lowercase, so nothing
    triggers it today -- but it is a divergence in the contract, recorded here rather than
    left to be discovered by the first capitalised log line.
    """
    output = "[whisper-prebuilt] existing whisper.cpp install Already Matches v1.9.1-unsloth.1"
    sh = _run_bash(_sh_whisper_script(tmp_path, 0, output))
    ps = _run_ps1(tmp_path, _ps1_whisper_script(tmp_path, 0, output))
    assert _steps(sh.stdout).get("whisper.cpp") == "prebuilt installed", sh.stdout
    assert _steps(ps.stdout).get("whisper.cpp") == "prebuilt up to date", ps.stdout


@requires_bash
def test_setup_sh_routes_the_remaining_whisper_exit_codes(tmp_path):
    """Exit 2 and exit 3 keep their own labels; neither is the new keep arm."""
    busy = _run_bash(_sh_whisper_script(tmp_path, 3, "[whisper-prebuilt] install busy"))
    assert _steps(busy.stdout)["whisper.cpp"] == "install busy; keeping existing runtime"

    skew = _run_bash(
        _sh_whisper_script(tmp_path, 2, "[whisper-prebuilt] slim bundle requires llama.cpp b9999;")
    )
    label = _steps(skew.stdout)["whisper.cpp"]
    assert label.startswith("no compatible prebuilt ("), label
    assert "b9999" in label, label
    assert "existing prebuilt kept" not in label, label


@requires_pwsh
def test_setup_ps1_routes_the_remaining_whisper_exit_codes(tmp_path):
    busy = _run_ps1(tmp_path, _ps1_whisper_script(tmp_path, 3, "[whisper-prebuilt] install busy"))
    assert _steps(busy.stdout)["whisper.cpp"] == "install busy; keeping existing runtime"

    skew = _run_ps1(
        tmp_path,
        _ps1_whisper_script(
            tmp_path, 2, "[whisper-prebuilt] slim bundle requires llama.cpp b9999;"
        ),
        name = "skew.ps1",
    )
    label = _steps(skew.stdout)["whisper.cpp"]
    assert label.startswith("no compatible prebuilt ("), label
    assert "b9999" in label, label
    assert "existing prebuilt kept" not in label, label


_LLAMA_KEEP_LOG = (
    "[llama-prebuilt] llama.cpp update unavailable, existing prebuilt kept; keeping the "
    "existing complete install of b1000"
)


@requires_bash
def test_setup_sh_reports_each_component_separately(tmp_path):
    """llama keeps its install while whisper's update fails: two labels, no cross-talk.

    Both blocks read from their own log file and their own status variable, and llama's
    keep arm must not leak into whisper's line (or the reverse) just because the same
    token appears in the same setup run.
    """
    llama_log = tmp_path / "prebuilt.log"
    llama_log.write_text(_LLAMA_KEEP_LOG + "\n", encoding = "utf-8")
    whisper_log = tmp_path / "whisper.log"
    whisper_log.write_text(
        "[whisper-prebuilt] prebuilt install failed: could not fetch release\n", encoding = "utf-8"
    )
    script = "\n".join(
        [
            _SH_HARNESS,
            f'SCRIPT_DIR="{tmp_path}"',
            f'UNSLOTH_HOME="{tmp_path}"',
            f'LLAMA_CPP_DIR="{tmp_path / "llama.cpp"}"',
            f'WHISPER_CPP_DIR="{tmp_path / "whisper.cpp"}"',
            f'_PREBUILT_LOG="{llama_log}"',
            "_PREBUILT_STATUS=0",
            '_explicit_llama_source_backend=""',
            _sh_block("llama"),
            f'_WHISPER_LOG="{whisper_log}"',
            "_WHISPER_STATUS=1",
            _sh_block("whisper"),
        ]
    )
    result = _run_bash(script)
    assert result.returncode == 0, result
    steps = _steps(result.stdout)
    assert steps["llama.cpp"] == "update unavailable, existing prebuilt kept", steps
    assert steps["whisper.cpp"].startswith(_FAILURE_LABEL_PREFIX), steps
    # A failed whisper never aborts setup: dictation is fail-open.
    assert "setup_fail|" not in result.stdout, result.stdout


@requires_pwsh
def test_setup_ps1_reports_each_component_separately(tmp_path):
    script = "\n".join(
        [
            _PS1_HARNESS,
            f'$LlamaCppDir = "{tmp_path / "llama.cpp"}"',
            f'$WhisperCppDir = "{tmp_path / "whisper.cpp"}"',
            "$prebuiltExit = 0",
            "$prebuiltOutput = @'",
            _LLAMA_KEEP_LOG,
            "'@",
            _ps1_block("llama"),
            "$whisperExit = 1",
            "$whisperOutput = @'",
            "[whisper-prebuilt] prebuilt install failed: could not fetch release",
            "'@",
            _ps1_block("whisper"),
        ]
    )
    result = _run_ps1(tmp_path, script, name = "both.ps1")
    assert result.returncode == 0, (result.stdout, result.stderr)
    steps = _steps(result.stdout)
    assert steps["llama.cpp"] == "update unavailable, existing prebuilt kept", steps
    assert steps["whisper.cpp"].startswith(_FAILURE_LABEL_PREFIX), steps
    assert "setup_fail|" not in result.stdout, result.stdout


@requires_bash
def test_the_installer_emits_exactly_the_substrings_the_scripts_grep(tmp_path, monkeypatch, capsys):
    """End to end: the real installer log, fed to the real setup.sh block.

    Every other test in this file asserts one half. This one runs the installer offline over
    an intact install, takes whatever it printed, and hands it to the shipped shell block --
    so a reworded log line shows up as the wrong user-facing label, not as a passing test.
    """
    host = _host("linux", "x64")
    install_dir = _seed_install(tmp_path, host)
    _inject(monkeypatch, "offline")
    code, log = _run_cli(monkeypatch, capsys, host, install_dir)
    assert code == M.EXIT_SUCCESS, log

    result = _run_bash(_sh_whisper_script(tmp_path, code, log))
    assert (
        _steps(result.stdout)["whisper.cpp"] == "update unavailable, existing prebuilt kept"
    ), result.stdout


# An untrustworthy release is an ANSWER, not an unavailability
# The keep arm exists because a lookup that could not answer says nothing about the tree on disk.
# A manifest digest disagreeing with the checksum index is the opposite: the release was fetched
# and found untrustworthy. Reporting "update unavailable, existing prebuilt kept" over it turns a
# tamper signal into a routine offline notice, and setup then paints it yellow rather than red.
# Unit coverage of the exception TYPE is not enough here: with the re-raise removed the whole
# install suite stayed green and only an end-to-end run noticed, which is why this drives the CLI.
def test_a_tampered_release_is_not_reported_as_an_unavailable_update(tmp_path, monkeypatch, capsys):
    host = _host("linux", "x64")
    install_dir = _seed_install(tmp_path, host)
    before = _tree_snapshot(install_dir)

    def tampered(*_a, **_k):
        raise M.core.ReleaseIntegrityError(
            "manifest sha256 for whisper-x.tar.gz disagrees with whisper-prebuilt-sha256.json; "
            "refusing a possibly tampered release"
        )

    monkeypatch.setattr(M, "_release_plan_for_host", tampered)
    code, log = _run_cli(monkeypatch, capsys, host, install_dir)

    assert code != M.EXIT_SUCCESS, f"a tampered release reported success\n{log}"
    assert KEEP_TOKEN not in log, f"the keep arm swallowed an integrity failure\n{log}"
    assert "tampered" in log, f"the reason never reached the user\n{log}"
    # Refusing to bless it must not damage the install that is already there.
    assert _tree_snapshot(install_dir) == before, "the intact tree moved"


def test_an_unavailable_lookup_still_keeps_the_install(tmp_path, monkeypatch, capsys):
    """The other half of the same fork, so the fix above cannot be 'refuse everything'."""
    host = _host("linux", "x64")
    install_dir = _seed_install(tmp_path, host)

    def unavailable(*_a, **_k):
        raise M.PrebuiltFallback("could not fetch release: network is unreachable")

    monkeypatch.setattr(M, "_release_plan_for_host", unavailable)
    code, log = _run_cli(monkeypatch, capsys, host, install_dir)

    assert code == M.EXIT_SUCCESS, f"an unavailable lookup should keep the install\n{log}"
    assert KEEP_TOKEN in log, log
