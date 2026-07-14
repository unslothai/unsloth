# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the prebuilt sd-cli asset resolver (``install_sd_cpp_prebuilt``).

Pure: the host -> release-asset matrix is exercised against a fixed asset list
(a real stable-diffusion.cpp release), no network. The installer lives under
``studio/`` (not ``studio/backend``), so the test puts that dir on the path.
"""

from __future__ import annotations

import sys
from pathlib import Path

_STUDIO = Path(__file__).resolve().parents[2]
if str(_STUDIO) not in sys.path:
    sys.path.insert(0, str(_STUDIO))

import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import urllib.error  # noqa: E402
import zipfile  # noqa: E402

import pytest  # noqa: E402

import install_sd_cpp_prebuilt as sdmod  # noqa: E402
from install_sd_cpp_prebuilt import (  # noqa: E402
    DEFAULT_REPO,
    DEFAULT_TAG,
    _fetch_release,
    _pinned_tag,
    _repo,
    _safe_extractall,
    _verify_sha256,
    default_install_dir,
    install,
    resolve_release_asset,
)

# A real stable-diffusion.cpp latest-release asset list.
_ASSETS = [
    "cudart-sd-bin-win-cu12-x64.zip",
    "sd-master-8caa3f9-bin-Darwin-macOS-15.7.7-arm64.zip",
    "sd-master-8caa3f9-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.13.0.zip",
    "sd-master-8caa3f9-bin-Linux-Ubuntu-24.04-x86_64-rocm-7.2.1.zip",
    "sd-master-8caa3f9-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip",
    "sd-master-8caa3f9-bin-Linux-Ubuntu-24.04-x86_64.zip",
    "sd-master-8caa3f9-bin-win-avx-x64.zip",
    "sd-master-8caa3f9-bin-win-avx2-x64.zip",
    "sd-master-8caa3f9-bin-win-avx512-x64.zip",
    "sd-master-8caa3f9-bin-win-cuda12-x64.zip",
    "sd-master-8caa3f9-bin-win-noavx-x64.zip",
    "sd-master-8caa3f9-bin-win-rocm-7.13.0-x64.zip",
    "sd-master-8caa3f9-bin-win-vulkan-x64.zip",
]


def _resolve(
    system,
    machine,
    accelerator = "auto",
):
    return resolve_release_asset(_ASSETS, system = system, machine = machine, accelerator = accelerator)


# ── macOS (the key Apple-Silicon target) ────────────────────────────────────


def test_macos_arm64_picks_darwin_arm64():
    assert _resolve("Darwin", "arm64") == "sd-master-8caa3f9-bin-Darwin-macOS-15.7.7-arm64.zip"
    # aarch64 spelling resolves the same
    assert _resolve("Darwin", "aarch64").startswith("sd-master") and "arm64" in _resolve(
        "Darwin", "aarch64"
    )


def test_macos_intel_has_no_prebuilt():
    # only an arm64 Darwin asset exists -> Intel Macs must build from source
    assert _resolve("Darwin", "x86_64") is None


# ── Linux (CPU is the default tier) ─────────────────────────────────────────


def test_linux_x86_64_auto_picks_plain_cpu_build():
    # the plain x86_64 zip, NOT a rocm/vulkan one
    assert _resolve("Linux", "x86_64") == "sd-master-8caa3f9-bin-Linux-Ubuntu-24.04-x86_64.zip"


def test_linux_vulkan_and_rocm_select_accelerator_builds():
    assert (
        _resolve("Linux", "x86_64", "vulkan")
        == "sd-master-8caa3f9-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip"
    )
    assert "rocm" in _resolve("Linux", "x86_64", "rocm")


def test_linux_arm64_has_no_prebuilt():
    assert _resolve("Linux", "aarch64") is None


# ── Windows ─────────────────────────────────────────────────────────────────


def test_windows_auto_picks_avx2():
    assert _resolve("Windows", "AMD64") == "sd-master-8caa3f9-bin-win-avx2-x64.zip"


def test_windows_cuda_picks_cuda12():
    assert _resolve("Windows", "AMD64", "cuda") == "sd-master-8caa3f9-bin-win-cuda12-x64.zip"


def test_windows_vulkan_picks_vulkan():
    assert _resolve("Windows", "AMD64", "vulkan") == "sd-master-8caa3f9-bin-win-vulkan-x64.zip"


# ── cudart helper archive is never chosen as the engine ─────────────────────


def test_cudart_runtime_archive_never_selected():
    for accel in ("auto", "cuda", "vulkan", "rocm"):
        chosen = _resolve("Windows", "AMD64", accel)
        assert chosen is None or not chosen.startswith("cudart")


# ── install dir ─────────────────────────────────────────────────────────────


def test_default_install_dir_is_sibling_of_llama(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    d = default_install_dir()
    assert d.name == "stable-diffusion.cpp"
    assert d.parent.name == ".unsloth"


# ── version pin + source repo (reproducibility) ─────────────────────────────


def test_pinned_tag_default_and_override(monkeypatch):
    monkeypatch.delenv("UNSLOTH_SD_CPP_TAG", raising = False)
    assert _pinned_tag() == DEFAULT_TAG  # pinned, not "latest"
    monkeypatch.setenv("UNSLOTH_SD_CPP_TAG", "master-999-deadbee")
    assert _pinned_tag() == "master-999-deadbee"
    monkeypatch.setenv("UNSLOTH_SD_CPP_TAG", "")  # explicit empty -> track latest
    assert _pinned_tag() is None


def test_repo_default_and_override(monkeypatch):
    monkeypatch.delenv("UNSLOTH_SD_CPP_REPO", raising = False)
    # Default is the Unsloth mirror; the env override can point back to leejet upstream.
    assert _repo() == DEFAULT_REPO == "unslothai/stable-diffusion.cpp"
    monkeypatch.setenv("UNSLOTH_SD_CPP_REPO", "leejet/stable-diffusion.cpp")
    assert _repo() == "leejet/stable-diffusion.cpp"


# ── sha256 integrity check ──────────────────────────────────────────────────


def test_verify_sha256_accepts_matching_digest(tmp_path):
    f = tmp_path / "asset.zip"
    f.write_bytes(b"hello sd-cli")
    digest = "sha256:" + hashlib.sha256(b"hello sd-cli").hexdigest()
    _verify_sha256(f, digest)  # no raise


def test_verify_sha256_rejects_mismatch(tmp_path):
    f = tmp_path / "asset.zip"
    f.write_bytes(b"tampered")
    bad = "sha256:" + hashlib.sha256(b"original").hexdigest()
    with pytest.raises(RuntimeError, match = "sha256 mismatch"):
        _verify_sha256(f, bad)


def test_verify_sha256_skips_when_absent_or_unknown(tmp_path):
    f = tmp_path / "asset.zip"
    f.write_bytes(b"x")
    _verify_sha256(f, None)  # no digest published -> warn + proceed (no raise)
    _verify_sha256(f, "md5:abc")  # unrecognised algo -> skip (no raise)


# ── _fetch_release: pinned-tag 404 -> latest fallback ───────────────────────


def test_fetch_release_falls_back_to_latest_on_404(monkeypatch):
    calls: list[str] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"tag_name": "latest-xyz", "assets": []}).encode()

    def fake_urlopen(req, timeout = 30.0):
        url = getattr(req, "full_url", req)
        calls.append(url)
        if "/tags/" in url:
            raise urllib.error.HTTPError(url, 404, "not found", None, None)
        return _Resp()

    monkeypatch.setattr(sdmod.urllib.request, "urlopen", fake_urlopen)
    rel = _fetch_release("gone-tag", repo = "leejet/stable-diffusion.cpp")
    assert rel["tag_name"] == "latest-xyz"
    assert any("/tags/gone-tag" in c for c in calls) and any(c.endswith("/latest") for c in calls)


def test_fetch_release_propagates_non_404(monkeypatch):
    def fake_urlopen(req, timeout = 30.0):
        url = getattr(req, "full_url", req)
        raise urllib.error.HTTPError(url, 403, "rate limited", None, None)

    monkeypatch.setattr(sdmod.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        _fetch_release("any-tag")


# ── install(): download -> verify -> extract -> locate (offline) ────────────


def _zip_with_sd_cli() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("build/bin/sd-cli", b"#!/bin/sh\necho sd-cli\n")
    return buf.getvalue()


def _stub_release(monkeypatch, *, zip_bytes: bytes, digest: str):
    name = "sd-master-deadbee-bin-Linux-Ubuntu-24.04-x86_64.zip"
    release = {
        "tag_name": "master-1-deadbee",
        "assets": [
            {
                "name": name,
                "browser_download_url": f"https://example.invalid/{name}",
                "digest": digest,
            }
        ],
    }
    monkeypatch.setattr(sdmod, "_fetch_release", lambda *a, **k: release)
    monkeypatch.setattr(sdmod, "_download", lambda url, dest, **k: dest.write_bytes(zip_bytes))
    monkeypatch.setattr(sdmod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sdmod.platform, "machine", lambda: "x86_64")
    return name


def test_install_downloads_verifies_extracts(tmp_path, monkeypatch):
    zb = _zip_with_sd_cli()
    name = _stub_release(
        monkeypatch, zip_bytes = zb, digest = "sha256:" + hashlib.sha256(zb).hexdigest()
    )
    sd_cli = install(install_dir = tmp_path)
    assert sd_cli.name == "sd-cli" and sd_cli.is_file()
    assert not (tmp_path / name).exists()  # archive cleaned up after extract
    # Ownership marker lets the uninstaller delete a Studio-installed sd.cpp beside a custom
    # root while keeping a user's own stable-diffusion.cpp checkout.
    assert (tmp_path / ".unsloth-studio-owned").is_file()


def test_install_into_empty_dir_claims_ownership(tmp_path, monkeypatch):
    # An empty (or freshly created) target may be adopted: the marker is written so the uninstaller
    # can later remove the Studio-installed tree.
    zb = _zip_with_sd_cli()
    _stub_release(monkeypatch, zip_bytes = zb, digest = "sha256:" + hashlib.sha256(zb).hexdigest())
    empty = tmp_path / "sdcpp"
    empty.mkdir()  # exists but empty
    install(install_dir = empty)
    assert (empty / ".unsloth-studio-owned").is_file()


def test_install_into_nonempty_unowned_dir_is_refused(tmp_path, monkeypatch):
    # A pre-existing, non-empty directory Studio did not create (e.g. a user's own checkout) must
    # not be extracted into; install() refuses up front and leaves it untouched.
    zb = _zip_with_sd_cli()
    _stub_release(monkeypatch, zip_bytes = zb, digest = "sha256:" + hashlib.sha256(zb).hexdigest())
    target = tmp_path / "stable-diffusion.cpp"
    target.mkdir()
    user_file = target / "USER_WORK"
    user_file.write_text("keep", encoding = "utf-8")

    with pytest.raises(RuntimeError, match = "not a Studio-managed directory"):
        install(install_dir = target)

    # The user's directory is left exactly as it was: file intact, no marker, nothing extracted.
    assert user_file.read_text(encoding = "utf-8") == "keep"
    assert not (target / ".unsloth-studio-owned").exists()
    assert list(target.iterdir()) == [user_file]


def test_reinstall_into_owned_dir_keeps_ownership(tmp_path, monkeypatch):
    # A directory that already carries our marker (a prior Studio install / upgrade) stays owned
    # even though it is now non-empty.
    zb = _zip_with_sd_cli()
    _stub_release(monkeypatch, zip_bytes = zb, digest = "sha256:" + hashlib.sha256(zb).hexdigest())
    target = tmp_path / "stable-diffusion.cpp"
    target.mkdir()
    (target / ".unsloth-studio-owned").touch()
    (target / "old-junk").write_text("x", encoding = "utf-8")

    install(install_dir = target)
    assert (target / ".unsloth-studio-owned").is_file()


def test_install_sha256_mismatch_raises_and_cleans_up(tmp_path, monkeypatch):
    zb = _zip_with_sd_cli()
    name = _stub_release(monkeypatch, zip_bytes = zb, digest = "sha256:" + "0" * 64)
    with pytest.raises(RuntimeError, match = "sha256 mismatch"):
        install(install_dir = tmp_path)
    assert not (tmp_path / name).exists()  # the finally: drops the bad archive


def test_partial_install_failure_is_reclaimed_on_retry(tmp_path, monkeypatch):
    # A crash AFTER extraction (here the post-extract cudart fetch raises) leaves the target
    # non-empty. Because ownership is marked BEFORE the partial writes, the retry recognises the
    # debris as ours and re-extracts instead of tripping the "not a Studio-managed directory"
    # refusal, so native install self-heals without the user manually deleting the directory.
    zb = _zip_with_sd_cli()
    _stub_release(monkeypatch, zip_bytes = zb, digest = "sha256:" + hashlib.sha256(zb).hexdigest())
    target = tmp_path / "sdcpp"

    state = {"failed": False}

    def flaky_cudart(*a, **k):
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("simulated interrupted post-extract step")

    monkeypatch.setattr(sdmod, "_maybe_fetch_windows_cudart", flaky_cudart)

    with pytest.raises(RuntimeError, match = "simulated interrupted"):
        install(install_dir = target)
    # The partial install left extracted files AND the ownership marker (written before the writes).
    assert (target / ".unsloth-studio-owned").is_file()
    assert any(target.iterdir())

    # The retry (cudart now succeeds) must NOT be refused; it re-extracts over the partial debris.
    sd_cli = install(install_dir = target)
    assert sd_cli.name == "sd-cli" and sd_cli.is_file()
    assert (target / ".unsloth-studio-owned").is_file()


# ── safe extraction (Zip-Slip guard) ─────────────────────────────────────────


def test_safe_extractall_rejects_path_traversal(tmp_path):
    target = tmp_path / "install"
    target.mkdir()
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sd-cli", b"ok")
        zf.writestr("../escape.txt", b"pwned")  # escapes the install dir
    with zipfile.ZipFile(archive) as zf:
        with pytest.raises(RuntimeError, match = "unsafe path"):
            _safe_extractall(zf, target)
    assert not (tmp_path / "escape.txt").exists()


def test_safe_extractall_extracts_normal_members(tmp_path):
    target = tmp_path / "install"
    target.mkdir()
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("build/bin/sd-cli", b"ok")
    with zipfile.ZipFile(archive) as zf:
        _safe_extractall(zf, target)
    assert (target / "build" / "bin" / "sd-cli").read_bytes() == b"ok"


def test_find_sd_cpp_binary_honors_studio_home(tmp_path, monkeypatch):
    # A binary installed under a custom Studio root must be discovered without also
    # setting UNSLOTH_SD_CPP_PATH (matches default_install_dir's env handling).
    from core.inference import sd_cpp_engine as eng

    monkeypatch.delenv("SD_CLI_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_SD_CPP_PATH", raising = False)
    studio_home = tmp_path / "studio_root"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    binary = tmp_path / "stable-diffusion.cpp" / "build" / "bin" / "sd-cli"
    binary.parent.mkdir(parents = True)
    binary.write_bytes(b"x")
    assert eng.find_sd_cpp_binary() == str(binary)


# ── Unsloth mirror: default source + the CPU/Apple asset set it publishes ─────

_TAG = "master-741-484baa4"
# Exactly what unslothai/stable-diffusion.cpp's CI publishes (CPU + Apple only; GPU
# hosts run diffusers and never reach the native installer).
_MIRROR_ASSETS = [
    f"sd-{_TAG}-bin-Darwin-macOS-arm64.zip",
    f"sd-{_TAG}-bin-Darwin-macOS-x86_64.zip",
    f"sd-{_TAG}-bin-Linux-Ubuntu-22.04-x86_64.zip",
    f"sd-{_TAG}-bin-Linux-Ubuntu-24.04-aarch64.zip",
    f"sd-{_TAG}-bin-win-cpu-x64.zip",
]


def _mresolve(
    system,
    machine,
    accelerator = "auto",
):
    return resolve_release_asset(
        _MIRROR_ASSETS, system = system, machine = machine, accelerator = accelerator
    )


def test_default_repo_is_the_unsloth_mirror():
    assert sdmod.DEFAULT_REPO == "unslothai/stable-diffusion.cpp"
    assert sdmod.UPSTREAM_FALLBACK_REPO == "leejet/stable-diffusion.cpp"


def test_mirror_matrix_resolves_every_cpu_apple_host():
    assert _mresolve("Darwin", "arm64") == f"sd-{_TAG}-bin-Darwin-macOS-arm64.zip"
    assert _mresolve("Darwin", "x86_64") == f"sd-{_TAG}-bin-Darwin-macOS-x86_64.zip"
    # WSL reports as Linux x86_64, so this also covers WSL.
    assert _mresolve("Linux", "x86_64") == f"sd-{_TAG}-bin-Linux-Ubuntu-22.04-x86_64.zip"
    assert _mresolve("Linux", "aarch64") == f"sd-{_TAG}-bin-Linux-Ubuntu-24.04-aarch64.zip"
    assert _mresolve("Windows", "AMD64") == f"sd-{_TAG}-bin-win-cpu-x64.zip"
    assert _mresolve("Windows", "AMD64", "cpu") == f"sd-{_TAG}-bin-win-cpu-x64.zip"


# ── mirror -> upstream fallback in install() ─────────────────────────────────


def _stub_two_repos(monkeypatch, *, mirror_serves, upstream_serves, zip_bytes, digest):
    """Stub _fetch_release to serve (or 404) per repo, so install()'s mirror->upstream
    fallback can be exercised without network. Host pinned to Linux x86_64."""
    monkeypatch.delenv("UNSLOTH_SD_CPP_REPO", raising = False)
    monkeypatch.delenv("UNSLOTH_SD_CPP_TAG", raising = False)
    mirror_name = f"sd-{_TAG}-bin-Linux-Ubuntu-22.04-x86_64.zip"
    upstream_name = "sd-master-741-484baa4-bin-Linux-Ubuntu-24.04-x86_64.zip"

    def _rel(name):
        return {
            "tag_name": _TAG,
            "assets": [
                {
                    "name": name,
                    "browser_download_url": f"https://example.invalid/{name}",
                    "digest": digest,
                }
            ],
        }

    def fake_fetch(
        tag = None,
        *,
        repo = None,
        token = None,
        timeout = 30.0,
        allow_latest = True,
    ):
        r = repo or sdmod.DEFAULT_REPO
        if r == sdmod.DEFAULT_REPO:
            if mirror_serves:
                return _rel(mirror_name)
            raise urllib.error.HTTPError(f"https://api/{r}", 404, "not found", None, None)
        if upstream_serves:
            return _rel(upstream_name)
        raise urllib.error.HTTPError(f"https://api/{r}", 404, "not found", None, None)

    monkeypatch.setattr(sdmod, "_fetch_release", fake_fetch)
    monkeypatch.setattr(sdmod, "_download", lambda url, dest, **k: dest.write_bytes(zip_bytes))
    monkeypatch.setattr(sdmod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sdmod.platform, "machine", lambda: "x86_64")
    return mirror_name, upstream_name


def test_install_uses_mirror_when_available(tmp_path, monkeypatch, capsys):
    zb = _zip_with_sd_cli()
    _stub_two_repos(
        monkeypatch,
        mirror_serves = True,
        upstream_serves = True,
        zip_bytes = zb,
        digest = "sha256:" + hashlib.sha256(zb).hexdigest(),
    )
    sd_cli = install(install_dir = tmp_path)
    assert sd_cli.name == "sd-cli" and sd_cli.is_file()
    assert "unslothai/stable-diffusion.cpp" in capsys.readouterr().out


def test_install_falls_back_to_upstream_when_mirror_missing(tmp_path, monkeypatch, capsys):
    zb = _zip_with_sd_cli()
    _stub_two_repos(
        monkeypatch,
        mirror_serves = False,
        upstream_serves = True,
        zip_bytes = zb,
        digest = "sha256:" + hashlib.sha256(zb).hexdigest(),
    )
    sd_cli = install(install_dir = tmp_path)
    assert sd_cli.name == "sd-cli" and sd_cli.is_file()
    captured = capsys.readouterr()
    # The repo-fallback diagnostic goes to stderr so --print-asset's stdout stays a single
    # asset line; the "source ..." install note still prints to stdout.
    assert "falling back to leejet/stable-diffusion.cpp" in captured.err
    assert "source leejet/stable-diffusion.cpp" in captured.out


def test_install_errors_when_neither_source_serves(tmp_path, monkeypatch):
    _stub_two_repos(
        monkeypatch, mirror_serves = False, upstream_serves = False, zip_bytes = b"", digest = ""
    )
    with pytest.raises(RuntimeError, match = "No prebuilt sd-cli"):
        install(install_dir = tmp_path)


# ── explicit GPU accelerator on a CPU-only mirror -> no CPU substitution ──────


def test_mirror_windows_gpu_accel_is_no_match_not_cpu():
    # The mirror ships only a CPU win zip. An explicit --accelerator cuda/vulkan/rocm must
    # NOT silently resolve to that CPU build; it returns None so install() falls back to
    # upstream, which does build the accelerated asset.
    for accel in ("cuda", "vulkan", "rocm"):
        assert _mresolve("Windows", "AMD64", accel) is None
    # auto / cpu still resolve to the CPU build.
    assert _mresolve("Windows", "AMD64", "cpu") == f"sd-{_TAG}-bin-win-cpu-x64.zip"


def test_mirror_linux_gpu_accel_is_no_match_not_cpu():
    for accel in ("cuda", "vulkan", "rocm"):
        assert _mresolve("Linux", "x86_64", accel) is None
    assert _mresolve("Linux", "x86_64", "cpu") == f"sd-{_TAG}-bin-Linux-Ubuntu-22.04-x86_64.zip"


def test_upstream_full_matrix_still_resolves_gpu_accel():
    # Regression guard: the "explicit GPU accel -> None on miss" change must not break
    # the upstream matrix, which does publish the accelerated builds.
    assert _resolve("Windows", "AMD64", "cuda") == "sd-master-8caa3f9-bin-win-cuda12-x64.zip"
    assert _resolve("Linux", "x86_64", "vulkan").endswith("x86_64-vulkan.zip")
    assert "rocm" in _resolve("Linux", "x86_64", "rocm")


# ── explicit repo override suppresses the upstream fallback ───────────────────


def test_explicit_repo_override_equal_to_default_suppresses_fallback(tmp_path, monkeypatch):
    # A user who pins UNSLOTH_SD_CPP_REPO (even to the default value) must get exactly that
    # repo -- no surprise leejet substitution -- so a missing release errors instead.
    _stub_two_repos(
        monkeypatch, mirror_serves = False, upstream_serves = True, zip_bytes = b"", digest = ""
    )
    monkeypatch.setenv("UNSLOTH_SD_CPP_REPO", sdmod.DEFAULT_REPO)
    with pytest.raises(RuntimeError, match = "No prebuilt sd-cli"):
        install(install_dir = tmp_path)


# ── pinned tag missing on mirror -> pinned upstream before mirror latest ──────


def test_pinned_tag_prefers_upstream_pin_over_mirror_latest(tmp_path, monkeypatch, capsys):
    # The mirror lacks the pinned tag but has a newer latest; upstream has the pin. The
    # pinned upstream build must win over the unpinned mirror-latest (reproducibility).
    monkeypatch.delenv("UNSLOTH_SD_CPP_REPO", raising = False)
    monkeypatch.setenv("UNSLOTH_SD_CPP_TAG", "master-999-pinned")
    zb = _zip_with_sd_cli()
    digest = "sha256:" + hashlib.sha256(zb).hexdigest()

    def _rel(name, tag):
        return {
            "tag_name": tag,
            "assets": [
                {
                    "name": name,
                    "browser_download_url": f"https://example.invalid/{name}",
                    "digest": digest,
                }
            ],
        }

    mirror_latest = "sd-master-000-latest-bin-Linux-Ubuntu-22.04-x86_64.zip"
    upstream_pinned = "sd-master-999-pinned-bin-Linux-Ubuntu-24.04-x86_64.zip"

    def fake_fetch(
        tag = None,
        *,
        repo = None,
        token = None,
        timeout = 30.0,
        allow_latest = True,
    ):
        r = repo or sdmod.DEFAULT_REPO
        if r == sdmod.DEFAULT_REPO:
            if tag == "master-999-pinned":  # mirror lacks the pin
                if not allow_latest:
                    return None
                return _rel(mirror_latest, "master-000-latest")
            return _rel(mirror_latest, "master-000-latest")
        # upstream HAS the pin
        if tag == "master-999-pinned":
            return _rel(upstream_pinned, "master-999-pinned")
        return _rel(upstream_pinned, "master-999-pinned")

    monkeypatch.setattr(sdmod, "_fetch_release", fake_fetch)
    monkeypatch.setattr(sdmod, "_download", lambda url, dest, **k: dest.write_bytes(zb))
    monkeypatch.setattr(sdmod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sdmod.platform, "machine", lambda: "x86_64")

    install(install_dir = tmp_path)
    out = capsys.readouterr().out
    assert "source leejet/stable-diffusion.cpp release master-999-pinned" in out


# ── --print-asset routes through the primary/upstream fallback ────────────────


def test_print_asset_uses_upstream_fallback(monkeypatch, capsys):
    # A host the mirror does not build (Linux Vulkan) must print the upstream asset that a
    # real install would fetch, not "no matching prebuilt".
    monkeypatch.delenv("UNSLOTH_SD_CPP_REPO", raising = False)
    monkeypatch.delenv("UNSLOTH_SD_CPP_TAG", raising = False)

    def fake_fetch(
        tag = None,
        *,
        repo = None,
        token = None,
        timeout = 30.0,
        allow_latest = True,
    ):
        r = repo or sdmod.DEFAULT_REPO
        if r == sdmod.DEFAULT_REPO:
            return {"tag_name": _TAG, "assets": [{"name": n} for n in _MIRROR_ASSETS]}
        return {"tag_name": "master-8caa3f9", "assets": [{"name": n} for n in _ASSETS]}

    monkeypatch.setattr(sdmod, "_fetch_release", fake_fetch)
    monkeypatch.setattr(sdmod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sdmod.platform, "machine", lambda: "x86_64")
    rc = sdmod.main(["--print-asset", "--accelerator", "vulkan"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "vulkan" in out and "no matching prebuilt" not in out
