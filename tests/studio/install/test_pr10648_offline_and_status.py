# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An offline lookup must keep an intact install and never report success over nothing installed."""

from __future__ import annotations

import email.message
import io
import json
import shutil
import socket
import subprocess
import tarfile
import urllib.error
from pathlib import Path

import pytest

from _pr10648_helpers import PACKAGE_ROOT, git, load_studio_module


MODULE_PATH = PACKAGE_ROOT / "studio" / "install_whisper_prebuilt.py"

# Distinct sys.modules name, or per-worker caches share monkeypatches.
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

# The substrings the setup scripts grep.
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
    """Seeds a real prebuilt tree and marker under its own MonkeyPatch; the injected failure stays live."""
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


FAILURE_MODES = (
    "offline",  # urllib.error.URLError: no network at all
    "timeout",  # socket.timeout on both the CDN HEAD and every GET
    "http_403",  # api.github.com rate limit -> fetch_json raises RuntimeError
    "http_429",  # api.github.com secondary rate limit -> same RuntimeError
    "http_404",
    "http_500",
    "truncated_json",  # a body that stops mid-object
    "malformed_json",  # a captive-portal/proxy HTML page where JSON was expected
    "empty_releases",
)


class _InjectedNetwork:
    """Replaces download_bytes and _URL_OPENER with one injected failure; calls shows it fired."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls: list[tuple[str, str]] = []
        self.downloads: list[str] = []

    def _raise_transport(self, url: str) -> None:
        if self.mode == "offline":
            raise urllib.error.URLError(OSError("injected: network is unreachable"))
        if self.mode == "timeout":
            raise socket.timeout("injected: timed out")

    def open(
        self,
        request,
        timeout = None,
    ):
        url = getattr(request, "full_url", str(request))
        self.calls.append(("head", url))
        self._raise_transport(url)
        # A 404 is how download_host_latest_release_tag says 'cannot name the release': None, not a raise.
        raise _http_error(404, url)

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

    def llama_latest_tag(self, repo: str) -> str | None:
        self.calls.append(("precheck", repo))
        self._raise_transport(repo)
        raise _http_error(404, f"https://github.com/{repo}/releases/latest")

    def download_file(self, url, destination):  # pragma: no cover - asserted never to run
        self.downloads.append(str(url))
        raise AssertionError(f"a keep/fail path must not download anything, but fetched {url}")

    @property
    def install_calls(self) -> list[tuple[str, str]]:
        """Install-path calls only; a fully offline run never reaches download_bytes at all."""
        return [(kind, url) for kind, url in self.calls if kind != "precheck"]


def _inject(monkeypatch, mode: str) -> _InjectedNetwork:
    """Patch both modules' primitives and prebuilt_core._URL_OPENER; one alias alone intercepts nothing."""
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


@pytest.mark.parametrize("mode", FAILURE_MODES)
def test_every_lookup_failure_keeps_an_intact_install(tmp_path, monkeypatch, capsys, mode):
    """An unpinned update whose release lookup fails keeps the intact install and exits 0."""
    host = _host("linux", "x64")
    install_dir = _seed_install(tmp_path, host)
    before = _tree_snapshot(install_dir)
    marker_before = M.metadata_path(install_dir).read_bytes()

    net = _inject(monkeypatch, mode)
    code, log = _run_cli(monkeypatch, capsys, host, install_dir)

    assert net.install_calls, f"{mode}: the injection never fired; the test intercepted nothing"
    assert code == M.EXIT_SUCCESS, f"{mode}: exit {code}\n{log}"
    # The substring setup.sh and setup.ps1 grep for.
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
    """Every failure mode and damage shape must exit non-zero with 'prebuilt install failed'."""
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
    """An explicitly pinned release must fail rather than keep a different installed release."""
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
    """--force, --whisper-tag and UNSLOTH_WHISPER_FORCE_COMPILE=1 ask for work that keeping would skip."""
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
    # The 403 is fetch_json's RuntimeError, not OSError; only the wrap makes it a PrebuiltFallback.
    assert "403" in log, log


def test_a_release_compatibility_error_is_never_papered_over(tmp_path, monkeypatch, capsys):
    """Exit 2 means the lookup answered with no compatible bundle; it must never become a kept success."""
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
    """When the release listing fails, the newest release's incompatibility in first_error is discarded."""
    newer_dir = tmp_path / "newer"
    newer_dir.mkdir(parents = True, exist_ok = True)
    archive, asset, sha256 = _build_cpu_bundle(newer_dir, host)
    newest = _bundle(host, asset, sha256, min_os = "26.0")
    # Control: the same bundle selects on a newer Mac, so the rejection is the min_os floor.
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
    """Unreachable listing rules nothing out, so keeping a min_os-compatible install is the right
    verdict."""
    host = _host("macos", "arm64", macos_version = (13, 0))
    install_dir = _seed_install(tmp_path, host, min_os = "13.0")
    before = _tree_snapshot(install_dir)

    net = _macos_listing_failure(tmp_path, monkeypatch, host)
    code, log = _run_cli(monkeypatch, capsys, host, install_dir)

    # This is the call whose failure discards first_error.
    assert [url for _kind, url in net.install_calls] == [
        "https://api.github.com/repos/unslothai/whisper.cpp/releases?per_page=100"
    ], net.install_calls
    assert code == M.EXIT_SUCCESS, f"exit {code}\n{log}"
    assert KEEP_TOKEN in log, log
    assert "could not list" in log, log
    assert "no whisper.cpp prebuilt asset" not in log, log
    assert _tree_snapshot(install_dir) == before


def test_macos_incompatible_newest_plus_unreachable_listing_fails_without_an_install(
    tmp_path, monkeypatch, capsys
):
    """With no install and an unreachable listing, the run must exit 1 rather than report a false
    success."""
    host = _host("macos", "arm64", macos_version = (13, 0))
    install_dir = tmp_path / "whisper.cpp"

    net = _macos_listing_failure(tmp_path, monkeypatch, host)
    code, log = _run_cli(monkeypatch, capsys, host, install_dir)

    assert net.install_calls
    assert code == M.EXIT_ERROR, f"exit {code}\n{log}"
    assert KEEP_TOKEN not in log, log
    assert FAIL_TOKEN in log, log


def test_macos_keep_refuses_an_install_below_this_hosts_floor(tmp_path, monkeypatch, capsys):
    """An install built for a newer macOS must not be kept offline on an older Mac; it cannot load there."""
    build_host = _host("macos", "arm64", macos_version = (15, 0))
    install_dir = _seed_install(tmp_path, build_host, min_os = "15.0")
    older_mac = _host("macos", "arm64", macos_version = (13, 0))

    # Assert the tree is intact on its own Mac so the exit 1 cannot pass for another reason.
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

    assert (
        "    for release_tag in _published_release_tags(published_repo):" in source
    ), "the pre-PR planner did not iterate the listing directly; re-read the comparison"
    assert "could not list" not in source, "pre-PR already had the listing guard"
    pre_install = source[source.index("def install_prebuilt(") :]
    pre_install = pre_install[: pre_install.index("\ndef ")]
    assert "except PrebuiltFallback" not in pre_install, pre_install
    assert KEEP_TOKEN not in source, "pre-PR already logged the keep token"

    post = MODULE_PATH.read_text(encoding = "utf-8")
    assert "except (OSError, RuntimeError) as exc:" in post
    guard = 'raise PrebuiltFallback(f"could not list {published_repo} releases: {exc}") from exc'
    assert guard in post


def test_the_keep_arm_sits_beside_the_status_arms_it_did_not_replace():
    """Keep arm must sit beside existing status arms, one per runtime per script, without rewriting them."""
    setup_sh = (PACKAGE_ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")
    setup_ps1 = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")

    # One keep arm per runtime per script, each in its own shell's syntax.
    sh_keeps = [line for line in setup_sh.splitlines() if KEEP_TOKEN in line]
    ps1_keeps = [line for line in setup_ps1.splitlines() if KEEP_TOKEN in line]
    # llama.cpp and audio.cpp; whisper.cpp is retired and setup no longer runs it.
    assert len(sh_keeps) == 2, sh_keeps
    assert len(ps1_keeps) == 2, ps1_keeps
    assert all("grep -Fq" in line for line in sh_keeps), sh_keeps
    assert all("-match" in line for line in ps1_keeps), ps1_keeps

    assert 'grep -Fq "already matches" "$_PREBUILT_LOG"' in setup_sh
    assert 'grep -Fq "already matches" "$_NODE_LOG"' in setup_sh
    assert f'$prebuiltOutput -match "{MATCH_TOKEN}"' in setup_ps1
    for script in (setup_sh, setup_ps1):
        assert (
            script.count("update unavailable, existing prebuilt kept") == 2
        ), "llama and audio.cpp each have exactly one keep arm"


# A digest mismatch is a tamper signal, not an offline notice; only an end-to-end CLI run caught
# the removed re-raise.
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
