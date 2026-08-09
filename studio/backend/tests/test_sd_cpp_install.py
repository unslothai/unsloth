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
import types  # noqa: E402
import threading  # noqa: E402
import io  # noqa: E402
import re  # noqa: E402
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
    upstream_tag_for,
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


def test_windows_arm64_takes_no_x64_build():
    """An x64 zip cannot run natively on a Windows arm64 host. Reporting no match lets the caller
    fall back instead of downloading and installing a binary that fails on first launch."""
    for accel in ("auto", "cuda", "vulkan", "rocm"):
        assert _resolve("Windows", "ARM64", accel) is None
        assert _resolve("Windows", "aarch64", accel) is None


def test_windows_arm64_picks_an_arm64_build_when_one_exists():
    assets = [*_ASSETS, "sd-master-8caa3f9-bin-win-avx2-arm64.zip"]
    assert (
        resolve_release_asset(assets, system = "Windows", machine = "ARM64", accelerator = "auto")
        == "sd-master-8caa3f9-bin-win-avx2-arm64.zip"
    )


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
    # The ownership marker lets the uninstaller delete a Studio-installed sd.cpp while keeping a user's own checkout.
    assert (tmp_path / ".unsloth-studio-owned").is_file()


def test_install_into_empty_dir_claims_ownership(tmp_path, monkeypatch):
    # An empty (or freshly created) target may be adopted: the marker is written so the uninstaller can remove the tree later.
    zb = _zip_with_sd_cli()
    _stub_release(monkeypatch, zip_bytes = zb, digest = "sha256:" + hashlib.sha256(zb).hexdigest())
    empty = tmp_path / "sdcpp"
    empty.mkdir()  # exists but empty
    install(install_dir = empty)
    assert (empty / ".unsloth-studio-owned").is_file()


def test_install_into_nonempty_unowned_dir_is_refused(tmp_path, monkeypatch):
    # A pre-existing, non-empty directory Studio did not create must not be extracted into; install() refuses up front.
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
    # A directory that already carries our marker stays owned even though it is now non-empty.
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
    # A crash AFTER extraction leaves the target non-empty. Because ownership is marked BEFORE the partial writes, the retry
    # recognises the debris as ours and re-extracts instead of tripping the "not a Studio-managed directory" refusal.
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
    # The partial install left extracted files AND the ownership marker.
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
    # A binary installed under a custom Studio root must be discovered without also setting UNSLOTH_SD_CPP_PATH.
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

# The shipped pin, not a copy of it: a hardcoded tag here silently stops describing what users
# actually install the moment DEFAULT_TAG moves.
_TAG = DEFAULT_TAG
# Exactly what unslothai/stable-diffusion.cpp's CI publishes. It was CPU and Apple only, on the
# premise that a GPU host runs diffusers instead. MiniMax-H3 falsified that: its diffusers path
# wants more VRAM than a consumer card has, so those hosts fall back to the native engine, and on
# Linux there was no accelerated build to fall back to. The CUDA leg is best effort and outside the
# publisher's coverage gate, so it can be absent from a release; every test below has to hold
# either way.
_MIRROR_ASSETS = [
    f"sd-{_TAG}-bin-Darwin-macOS-arm64.zip",
    f"sd-{_TAG}-bin-Darwin-macOS-x86_64.zip",
    # Before the plain build, which is the order a real release lists them in ("-cuda12.zip"
    # sorts ahead of ".zip"). It matters: if auto picked by position rather than by the
    # accelerator-marker filter, this ordering is what would expose it.
    f"sd-{_TAG}-bin-Linux-Ubuntu-22.04-x86_64-cuda12.zip",
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


def test_mirror_linux_cuda_resolves_the_cuda_bundle():
    """A CUDA host asking for cuda gets the accelerated build. This is what makes MiniMax-H3
    usable there: its GGUF path was running on the CPU because no Linux CUDA asset existed, and
    video.py falls back to the CPU prebuilt whenever the accelerated one cannot be resolved."""
    assert (
        _mresolve("Linux", "x86_64", "cuda")
        == f"sd-{_TAG}-bin-Linux-Ubuntu-22.04-x86_64-cuda12.zip"
    )


def test_mirror_linux_auto_still_takes_the_plain_cpu_build():
    """The CUDA bundle ships the CUDA runtime and is roughly 25x the size of the CPU one. A host
    that did not ask for a GPU build must not be handed it by accident."""
    assert _mresolve("Linux", "x86_64") == f"sd-{_TAG}-bin-Linux-Ubuntu-22.04-x86_64.zip"
    assert _mresolve("Linux", "x86_64", "cpu") == f"sd-{_TAG}-bin-Linux-Ubuntu-22.04-x86_64.zip"


def test_mirror_linux_cuda_refuses_a_release_without_one():
    """The CUDA leg is best effort, so a release can lack the asset. Returning None lets the
    caller fall back deliberately; handing back the CPU build would leave the caller believing it
    had a GPU binary, keep the load on the GPU device, and run sd-cli wholly on the CPU."""
    without = [a for a in _MIRROR_ASSETS if "cuda" not in a]
    assert (
        resolve_release_asset(without, system = "Linux", machine = "x86_64", accelerator = "cuda") is None
    )


# ── the pin translates to upstream ───────────────────────────────────────────


def test_upstream_tag_drops_the_mirror_fork_suffix():
    """A mirror release built on an upstream one is that tag plus "-u<fork sha>", and only the
    base half exists upstream. The suffix has to come off before the fallback asks for it."""
    assert upstream_tag_for("master-813-bfbef5b-u13b9d92") == "master-813-bfbef5b"
    assert upstream_tag_for("master-813-bfbef5b-u0665242") == "master-813-bfbef5b"
    # A plain upstream tag, and a tag whose trailing segment is not a fork sha, pass through.
    assert upstream_tag_for("master-809-eb7f35c") == "master-809-eb7f35c"
    assert upstream_tag_for("v1.2.3-ubuntu") == "v1.2.3-ubuntu"
    assert upstream_tag_for(None) is None


def test_shipped_pin_translates_to_a_plain_upstream_tag():
    """Guards the pin itself: whatever DEFAULT_TAG becomes, the string handed upstream must not
    still carry a fork suffix, or the fallback is asking for a release that cannot exist."""
    assert re.search(r"-u[0-9a-f]{7,}$", upstream_tag_for(DEFAULT_TAG)) is None


def test_upstream_fallback_asks_for_the_translated_pin_not_latest(monkeypatch):
    """A Linux Vulkan host: the mirror has no such asset, so resolution falls to upstream. It must
    ask upstream for the pin's upstream base tag and stop there -- reaching an unpinned upstream
    "latest" is exactly the reproducibility loss the translation exists to prevent."""
    monkeypatch.delenv("UNSLOTH_SD_CPP_REPO", raising = False)
    monkeypatch.setenv("UNSLOTH_SD_CPP_TAG", "master-813-bfbef5b-u13b9d92")
    monkeypatch.setattr(sdmod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sdmod.platform, "machine", lambda: "x86_64")
    seen = []

    def fake_fetch(
        tag = None,
        *,
        repo = None,
        token = None,
        timeout = 30.0,
        allow_latest = True,
    ):
        seen.append((repo, tag))
        if repo == sdmod.UPSTREAM_FALLBACK_REPO and tag == "master-813-bfbef5b":
            return {
                "tag_name": tag,
                "assets": [{"name": "sd-master-bfbef5b-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip"}],
            }
        # The mirror serves the pin but builds no Vulkan asset; every other request 404s.
        if repo == sdmod.DEFAULT_REPO and tag == "master-813-bfbef5b-u13b9d92":
            return {
                "tag_name": tag,
                "assets": [{"name": f"sd-{tag}-bin-Linux-Ubuntu-22.04-x86_64.zip"}],
            }
        raise urllib.error.HTTPError(f"https://api/{repo}", 404, "not found", None, None)

    monkeypatch.setattr(sdmod, "_fetch_release", fake_fetch)
    repo, release, chosen = sdmod._resolve_with_fallback("vulkan", None)
    assert repo == sdmod.UPSTREAM_FALLBACK_REPO
    assert release["tag_name"] == "master-813-bfbef5b"
    assert chosen.endswith("-vulkan.zip")
    # The raw fork tag is never sent upstream, and no unpinned latest is ever requested.
    assert (sdmod.UPSTREAM_FALLBACK_REPO, "master-813-bfbef5b-u13b9d92") not in seen
    assert not any(tag is None for _repo_name, tag in seen)


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
    # The repo-fallback diagnostic goes to stderr so --print-asset's stdout stays one asset line; the install note stays on stdout.
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
    # The mirror ships only a CPU win zip. An explicit --accelerator cuda/vulkan/rocm must NOT silently resolve to it; it
    # returns None so install() falls back to upstream, which does build the accelerated asset.
    for accel in ("cuda", "vulkan", "rocm"):
        assert _mresolve("Windows", "AMD64", accel) is None
    # auto / cpu still resolve to the CPU build.
    assert _mresolve("Windows", "AMD64", "cpu") == f"sd-{_TAG}-bin-win-cpu-x64.zip"


def test_mirror_linux_gpu_accel_is_no_match_not_cpu():
    # cuda is no longer in this list: the mirror publishes a Linux CUDA bundle now, and
    # test_mirror_linux_cuda_resolves_the_cuda_bundle pins that. vulkan and rocm are still
    # unbuilt, and must return None rather than quietly resolving to the CPU zip.
    for accel in ("vulkan", "rocm"):
        assert _mresolve("Linux", "x86_64", accel) is None
    assert _mresolve("Linux", "x86_64", "cpu") == f"sd-{_TAG}-bin-Linux-Ubuntu-22.04-x86_64.zip"


def test_upstream_full_matrix_still_resolves_gpu_accel():
    # Regression guard: the "explicit GPU accel -> None on miss" change must not break the upstream matrix, which publishes them.
    assert _resolve("Windows", "AMD64", "cuda") == "sd-master-8caa3f9-bin-win-cuda12-x64.zip"
    assert _resolve("Linux", "x86_64", "vulkan").endswith("x86_64-vulkan.zip")
    assert "rocm" in _resolve("Linux", "x86_64", "rocm")


# ── explicit repo override suppresses the upstream fallback ───────────────────


def test_explicit_repo_override_equal_to_default_suppresses_fallback(tmp_path, monkeypatch):
    # A user who pins UNSLOTH_SD_CPP_REPO (even to the default) must get exactly that repo, so a missing release errors.
    _stub_two_repos(
        monkeypatch, mirror_serves = False, upstream_serves = True, zip_bytes = b"", digest = ""
    )
    monkeypatch.setenv("UNSLOTH_SD_CPP_REPO", sdmod.DEFAULT_REPO)
    with pytest.raises(RuntimeError, match = "No prebuilt sd-cli"):
        install(install_dir = tmp_path)


# ── pinned tag missing on mirror -> pinned upstream before mirror latest ──────


def test_pinned_tag_prefers_upstream_pin_over_mirror_latest(tmp_path, monkeypatch, capsys):
    # The mirror lacks the pinned tag but has a newer latest; the pinned upstream build must win (reproducibility).
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
    # A host the mirror does not build (Linux Vulkan) must print the upstream asset a real install would fetch.
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


def test_unrunnable_managed_binary_is_removed_so_it_reinstalls(monkeypatch, tmp_path):
    # An interrupted extraction leaves an sd-cli that exists but cannot run. The finder only checks is_file(), so without a
    # probe the installer never retried and native inference stayed off for the life of the install.
    import core.inference.sd_cpp_backend as bk
    import core.inference.sd_cpp_engine as eng

    root = tmp_path / "sd-home" / "stable-diffusion.cpp"
    root.mkdir(parents = True)
    # install() writes the ownership marker BEFORE extracting, so a real interrupted extraction always leaves one.
    (root / ".unsloth-studio-owned").touch()
    managed = root / "sd-cli"
    managed.write_bytes(b"truncated")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))
    assert eng.is_managed_binary(str(managed)) is True

    monkeypatch.setattr(
        bk, "find_sd_cpp_binary", lambda: str(managed) if managed.exists() else None
    )
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: False)
    installs: list = []

    def _install(**kwargs):
        installs.append(kwargs)
        managed.write_bytes(b"good")
        return managed

    import sys
    import types

    stub = types.ModuleType("install_sd_cpp_prebuilt")
    stub.install = _install
    monkeypatch.setitem(sys.modules, "install_sd_cpp_prebuilt", stub)

    out = bk.ensure_sd_cpp_binary(accelerator = "cpu")
    assert installs, "the unusable managed copy must trigger a reinstall"
    assert out == str(managed)


def test_an_unrunnable_user_supplied_binary_is_never_deleted(monkeypatch, tmp_path):
    # SD_CLI_PATH / PATH / an in-tree build belong to the user: report them and let the router's probe refuse, never remove them.
    import core.inference.sd_cpp_backend as bk

    outside = tmp_path / "mine" / "sd-cli"
    outside.parent.mkdir(parents = True)
    outside.write_bytes(b"truncated")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "elsewhere" / "studio"))
    monkeypatch.setattr(bk, "find_sd_cpp_binary", lambda: str(outside))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: False)

    assert bk.ensure_sd_cpp_binary(accelerator = "cpu") == str(outside)
    assert outside.exists(), "a user-supplied binary must survive"


def test_unrunnable_binary_in_an_unmarked_root_is_kept_because_install_would_refuse(
    monkeypatch, tmp_path
):
    # The repair and the installer must agree on what "ours" means: install() refuses a pre-existing, non-empty target with no marker,
    # which is what a user's own checkout looks like, so discarding a binary there deleted it and then refused the reinstall. Left in place.
    import types

    import core.inference.sd_cpp_backend as bk
    import core.inference.sd_cpp_engine as eng

    root = tmp_path / "sd-home" / "stable-diffusion.cpp" / "sd-bin"
    root.mkdir(parents = True)
    server = root / "sd-server"
    server.write_bytes(b"truncated")
    (root / "sd-cli").write_bytes(b"truncated")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))
    assert eng.is_managed_binary(str(server)) is False  # under our root, but unmarked

    monkeypatch.setattr(
        bk, "find_sd_server_binary", lambda: str(server) if server.exists() else None
    )
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: False)

    def _refuse(**_kwargs):
        raise RuntimeError("sd.cpp install target already exists and is not a Studio-managed dir")

    stub = types.ModuleType("install_sd_cpp_prebuilt")
    stub.install = _refuse
    monkeypatch.setitem(sys.modules, "install_sd_cpp_prebuilt", stub)

    assert bk.ensure_sd_server_binary(accelerator = "cpu") == str(server)
    assert server.is_file(), "an unmarked binary the installer cannot replace must survive"


def test_the_repair_only_deletes_what_the_installer_may_reinstall(monkeypatch, tmp_path):
    # The same tree WITH the marker is ours: install() reclaims it, so discarding the unrunnable copy is safe.
    import types

    import core.inference.sd_cpp_backend as bk

    root = tmp_path / "sd-home" / "stable-diffusion.cpp"
    (root / "sd-bin").mkdir(parents = True)
    (root / ".unsloth-studio-owned").touch()
    server = root / "sd-bin" / "sd-server"
    server.write_bytes(b"truncated")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))

    monkeypatch.setattr(
        bk, "find_sd_server_binary", lambda: str(server) if server.exists() else None
    )
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: False)
    installs: list = []

    def _install(**kwargs):
        installs.append(kwargs)
        server.write_bytes(b"good")
        return server

    stub = types.ModuleType("install_sd_cpp_prebuilt")
    stub.install = _install
    monkeypatch.setitem(sys.modules, "install_sd_cpp_prebuilt", stub)

    assert bk.ensure_sd_server_binary(accelerator = "cpu") == str(server)
    assert installs, "a marked (reclaimable) tree must still be repaired"


def test_a_reinstall_over_an_owned_root_keeps_the_repair_loop_closed(tmp_path, monkeypatch):
    # End to end across the two modules: after a real install() the marker exists, so the binary reads as managed and a later
    # repair may discard it. Without this the two definitions of "ours" drift apart again.
    import core.inference.sd_cpp_engine as eng

    zb = _zip_with_sd_cli()
    _stub_release(monkeypatch, zip_bytes = zb, digest = "sha256:" + hashlib.sha256(zb).hexdigest())
    target = tmp_path / "sd-home" / "stable-diffusion.cpp"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))

    sd_cli = install(install_dir = target)
    assert (target / ".unsloth-studio-owned").is_file()
    assert eng.is_managed_binary(str(sd_cli)) is True


# ── the install records its accelerator, and a change reinstalls ─────────────


def test_install_records_the_accelerator_it_installed(tmp_path, monkeypatch):
    """The record is what lets a later ensure_* tell a CPU bundle from a GPU one."""
    zb = _zip_with_sd_cli()
    _stub_release(monkeypatch, zip_bytes = zb, digest = "sha256:" + hashlib.sha256(zb).hexdigest())
    target = tmp_path / "sd-home" / "stable-diffusion.cpp"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))

    install(install_dir = target)
    assert sdmod.installed_accelerator(target) == "cpu"
    # "auto" resolves to the same plain build, so it must not read as a different install.
    assert sdmod.accelerator_class("auto") == sdmod.accelerator_class("cpu") == "cpu"
    assert sdmod.accelerator_class("CUDA") == "cuda"
    # No record at all is "unknown", not "cpu".
    assert sdmod.installed_accelerator(tmp_path / "nothing-here") is None


def _managed_tree(
    tmp_path,
    monkeypatch,
    accelerator = None,
):
    """A Studio-owned install tree, optionally carrying an install record."""
    root = tmp_path / "sd-home" / "stable-diffusion.cpp"
    (root / "sd-bin").mkdir(parents = True)
    (root / ".unsloth-studio-owned").touch()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))
    if accelerator is not None:
        sdmod._write_install_record(root, accelerator = accelerator, repo = "r", tag = "t")
    return root


def test_a_cpu_install_is_reinstalled_when_cuda_is_requested(tmp_path, monkeypatch):
    """The P1: an upgraded Linux CUDA host already holding the managed CPU bundle. Both ensure_*
    returned any runnable binary they found, so the new CUDA asset was never installed and native
    generation silently stayed entirely on the CPU."""
    import core.inference.sd_cpp_backend as bk

    root = _managed_tree(tmp_path, monkeypatch, accelerator = "cpu")
    server = root / "sd-bin" / "sd-server"
    server.write_bytes(b"cpu-build")
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(server))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    installs: list = []

    def _install(**kwargs):
        installs.append(kwargs)
        server.write_bytes(b"cuda-build")
        sdmod._write_install_record(root, accelerator = kwargs["accelerator"], repo = "r", tag = "t")
        return root / "sd-bin" / "sd-cli"

    monkeypatch.setattr(sdmod, "install", _install)

    assert bk.ensure_sd_server_binary(accelerator = "cuda") == str(server)
    assert [k["accelerator"] for k in installs] == ["cuda"], "the CUDA build must be installed"
    assert server.read_bytes() == b"cuda-build"
    # Now that the record says cuda, the next load reuses it instead of reinstalling.
    assert bk.ensure_sd_server_binary(accelerator = "cuda") == str(server)
    assert len(installs) == 1


def test_asking_for_the_cpu_build_never_reinstalls(tmp_path, monkeypatch):
    """Only an upgrade reinstalls. A CPU/auto request must reuse whatever is there, including an
    install predating the record, or every CPU host would re-download on its next load."""
    import core.inference.sd_cpp_backend as bk

    root = _managed_tree(tmp_path, monkeypatch, accelerator = None)  # no record: an older install
    server = root / "sd-bin" / "sd-server"
    server.write_bytes(b"cpu-build")
    cli = root / "sd-bin" / "sd-cli"
    cli.write_bytes(b"cpu-build")
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(server))
    monkeypatch.setattr(bk, "find_sd_cpp_binary", lambda: str(cli))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())

    def _install(**_kwargs):
        raise AssertionError("a CPU request must not reinstall")

    monkeypatch.setattr(sdmod, "install", _install)
    assert bk.ensure_sd_server_binary(accelerator = "cpu") == str(server)
    assert bk.ensure_sd_cpp_binary(accelerator = "auto") == str(cli)


def test_a_user_supplied_binary_is_never_reinstalled_over_on_an_accelerator_change(
    tmp_path, monkeypatch
):
    """An unmarked root is the user's own build. install() would refuse it anyway, so attempting
    the upgrade would only cost a download and end with no binary at all."""
    import core.inference.sd_cpp_backend as bk

    root = tmp_path / "sd-home" / "stable-diffusion.cpp"  # deliberately NOT marked
    (root / "sd-bin").mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "sd-home" / "studio"))
    server = root / "sd-bin" / "sd-server"
    server.write_bytes(b"users-own-build")
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(server))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())

    def _install(**_kwargs):
        raise AssertionError("an unmarked (user-owned) root must not be reinstalled over")

    monkeypatch.setattr(sdmod, "install", _install)
    assert bk.ensure_sd_server_binary(accelerator = "cuda") == str(server)


def test_a_failed_upgrade_keeps_the_working_binary_and_stops_retrying(tmp_path, monkeypatch):
    """A host with no asset for the requested accelerator must keep the binary it has (returning
    None would drop native inference entirely) and must not re-resolve on every later load."""
    import core.inference.sd_cpp_backend as bk

    root = _managed_tree(tmp_path, monkeypatch, accelerator = "cpu")
    server = root / "sd-bin" / "sd-server"
    server.write_bytes(b"cpu-build")
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(server))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    attempts: list = []

    def _install(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError("No prebuilt sd-cli for this host")

    monkeypatch.setattr(sdmod, "install", _install)

    assert bk.ensure_sd_server_binary(accelerator = "vulkan") == str(server)
    assert bk.ensure_sd_server_binary(accelerator = "vulkan") == str(server)
    assert len(attempts) == 1, "the hopeless upgrade must be attempted once, not once per load"
    assert server.read_bytes() == b"cpu-build"


def test_the_upgrade_waits_for_the_resident_server_to_stop(tmp_path, monkeypatch):
    """The install replaces the sd-server file, and the resident server is executing that exact
    path: Linux refuses to open a running executable for writing (ETXTBSY) and Windows locks it.
    Resolving must therefore NOT install while a server is up; the load retries once it is
    stopped, which is the only moment the file is free."""
    import core.inference.sd_cpp_backend as bk

    root = _managed_tree(tmp_path, monkeypatch, accelerator = "cpu")
    server = root / "sd-bin" / "sd-server"
    server.write_bytes(b"cpu-build")
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(server))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    monkeypatch.setattr(bk, "_install_allowed", lambda: True)
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(backend = "cuda")
    )
    installs: list = []

    def _install(**kwargs):
        installs.append(kwargs)
        server.write_bytes(b"cuda-build")
        sdmod._write_install_record(root, accelerator = kwargs["accelerator"], repo = "r", tag = "t")
        return root / "sd-bin" / "sd-cli"

    monkeypatch.setattr(sdmod, "install", _install)

    backend = bk.SdCppDiffusionBackend()
    # A resident server is up: resolving must hand back the existing binary and install nothing.
    backend._state = types.SimpleNamespace(server = object())
    mode, resolved, _engine = backend._resolve_backend()
    assert mode == "server" and resolved == str(server)
    assert installs == [], "no install may run while the server holds its own executable"
    assert server.read_bytes() == b"cpu-build"

    # Once it is stopped, the deferred upgrade lands.
    backend._state = None
    upgraded = backend._upgrade_server_after_teardown(str(server))
    assert [k["accelerator"] for k in installs] == ["cuda"]
    assert upgraded == str(server) and server.read_bytes() == b"cuda-build"
    # Now that the record matches, a later teardown does not reinstall again.
    assert backend._upgrade_server_after_teardown(str(server)) == str(server)
    assert len(installs) == 1


def test_a_failed_post_teardown_upgrade_keeps_the_existing_server(tmp_path, monkeypatch):
    """An upgrade may never cost the load the binary it already had."""
    import core.inference.sd_cpp_backend as bk

    root = _managed_tree(tmp_path, monkeypatch, accelerator = "cpu")
    server = root / "sd-bin" / "sd-server"
    server.write_bytes(b"cpu-build")
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(server))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    monkeypatch.setattr(bk, "_install_allowed", lambda: True)
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(backend = "cuda")
    )

    def _boom(**_kwargs):
        raise RuntimeError("no prebuilt for this host")

    monkeypatch.setattr(sdmod, "install", _boom)

    backend = bk.SdCppDiffusionBackend()
    backend._state = None
    assert backend._upgrade_server_after_teardown(str(server)) == str(server)
    assert server.read_bytes() == b"cpu-build"


def test_a_recorded_gpu_install_is_replaced_when_the_cpu_build_is_wanted(tmp_path, monkeypatch):
    """The mirror image of the CPU-to-CUDA upgrade. Nothing on the sd-server/sd-cli command line
    selects a backend -- the build itself is the choice -- so a recorded CUDA install keeps running
    on the GPU after the device target resolves to CPU. Only a RECORDED mismatch reinstalls: an
    unrecorded install stays put, else every legacy CPU host would redownload on a CPU target."""
    import core.inference.sd_cpp_backend as bk

    root = _managed_tree(tmp_path, monkeypatch, accelerator = "cuda")
    server = root / "sd-bin" / "sd-server"
    server.write_bytes(b"cuda-build")
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(server))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    installs: list = []

    def _install(**kwargs):
        installs.append(kwargs)
        server.write_bytes(b"cpu-build")
        sdmod._write_install_record(root, accelerator = kwargs["accelerator"], repo = "r", tag = "t")
        return root / "sd-bin" / "sd-cli"

    monkeypatch.setattr(sdmod, "install", _install)

    assert bk.ensure_sd_server_binary(accelerator = "cpu") == str(server)
    assert [k["accelerator"] for k in installs] == ["cpu"]
    assert server.read_bytes() == b"cpu-build"
    # The record now says cpu, so the next CPU load reuses it.
    assert bk.ensure_sd_server_binary(accelerator = "cpu") == str(server)
    assert len(installs) == 1


def test_the_upgrade_waits_for_an_active_one_shot_generation(tmp_path, monkeypatch):
    """A one-shot load holds the managed tree just as hard as a resident server: begin_load only
    signals the in-flight generation to cancel, and _resolve_backend runs before the load waits on
    _generate_lock, so the old sd-cli can still be executing from the tree an install would
    overwrite. Defer there too, and land the upgrade after the teardown."""
    import core.inference.sd_cpp_backend as bk

    root = _managed_tree(tmp_path, monkeypatch, accelerator = "cpu")
    server = root / "sd-bin" / "sd-server"
    server.write_bytes(b"cpu-build")
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(server))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    monkeypatch.setattr(bk, "_install_allowed", lambda: True)
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(backend = "cuda")
    )
    installs: list = []

    def _install(**kwargs):
        installs.append(kwargs)
        server.write_bytes(b"cuda-build")
        sdmod._write_install_record(root, accelerator = kwargs["accelerator"], repo = "r", tag = "t")
        return root / "sd-bin" / "sd-cli"

    monkeypatch.setattr(sdmod, "install", _install)

    backend = bk.SdCppDiffusionBackend()
    # One-shot: no resident server, but a generation is still running out of the tree.
    backend._state = types.SimpleNamespace(server = None)
    backend._active_generate_cancel = threading.Event()
    mode, resolved, _engine = backend._resolve_backend()
    assert mode == "server" and resolved == str(server)
    assert installs == [], "no install may run while a one-shot sd-cli is executing"
    assert server.read_bytes() == b"cpu-build"
    assert backend._deferred_accelerator_install is True

    # After the teardown the generation is over and the tree is free.
    backend._state = None
    backend._active_generate_cancel = None
    assert backend._upgrade_server_after_teardown(str(server)) == str(server)
    assert [k["accelerator"] for k in installs] == ["cuda"]
    assert server.read_bytes() == b"cuda-build"


def test_an_idle_backend_installs_the_matching_build_immediately(tmp_path, monkeypatch):
    """The deferral is only for a tree in use: with nothing running, resolving installs at once."""
    import core.inference.sd_cpp_backend as bk

    root = _managed_tree(tmp_path, monkeypatch, accelerator = "cpu")
    server = root / "sd-bin" / "sd-server"
    server.write_bytes(b"cpu-build")
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(server))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    monkeypatch.setattr(bk, "_install_allowed", lambda: True)
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(backend = "cuda")
    )
    installs: list = []

    def _install(**kwargs):
        installs.append(kwargs)
        server.write_bytes(b"cuda-build")
        sdmod._write_install_record(root, accelerator = kwargs["accelerator"], repo = "r", tag = "t")
        return root / "sd-bin" / "sd-cli"

    monkeypatch.setattr(sdmod, "install", _install)

    backend = bk.SdCppDiffusionBackend()
    mode, resolved, _engine = backend._resolve_backend()
    assert mode == "server" and resolved == str(server)
    assert [k["accelerator"] for k in installs] == ["cuda"]
    assert backend._deferred_accelerator_install is False


def test_the_router_entry_point_cannot_replace_a_running_server(tmp_path, monkeypatch):
    """select_and_activate_engine calls ensure_sd_server_binary DIRECTLY, before begin_load stops
    anything, so a deferral that lives only in _resolve_backend does not cover it: a /images/load
    that resolves to CUDA while the managed CPU server is resident would extract over the running
    executable. The refusal therefore lives in _accelerator_changed, where every caller passes."""
    import core.inference.sd_cpp_backend as bk

    root = _managed_tree(tmp_path, monkeypatch, accelerator = "cpu")
    server = root / "sd-bin" / "sd-server"
    server.write_bytes(b"cpu-build")
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(server))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    installs: list = []

    def _install(**kwargs):
        installs.append(kwargs)
        server.write_bytes(b"cuda-build")
        sdmod._write_install_record(root, accelerator = kwargs["accelerator"], repo = "r", tag = "t")
        return root / "sd-bin" / "sd-cli"

    monkeypatch.setattr(sdmod, "install", _install)

    resident = bk.SdCppDiffusionBackend()
    resident._state = types.SimpleNamespace(server = object())
    monkeypatch.setattr(bk, "_sd_cpp_backend", resident)

    # The router's own call, verbatim: no backend instance in sight.
    assert bk.ensure_sd_server_binary(accelerator = "cuda") == str(server)
    assert installs == [], "the running server's file may not be overwritten"
    assert server.read_bytes() == b"cpu-build"

    # With nothing resident the same call upgrades as before.
    monkeypatch.setattr(bk, "_sd_cpp_backend", None)
    assert bk.ensure_sd_server_binary(accelerator = "cuda") == str(server)
    assert [k["accelerator"] for k in installs] == ["cuda"]


def test_the_one_shot_fallback_keeps_the_requested_accelerator(tmp_path, monkeypatch):
    """_resolve_engine is also the fallback a GPU sd-server that would not start lands on. It
    asked for the default "cpu", which -- now that a recorded GPU install counts as a mismatch
    against a CPU request -- would reinstall the plain bundle over the working GPU one and run the
    whole generation on the CPU because of an unrelated server startup failure."""
    import core.inference.sd_cpp_backend as bk

    root = _managed_tree(tmp_path, monkeypatch, accelerator = "cuda")
    cli = root / "sd-bin" / "sd-cli"
    cli.write_bytes(b"cuda-build")
    monkeypatch.setattr(bk, "find_sd_cpp_binary", lambda: str(cli))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    monkeypatch.setattr(bk, "_install_allowed", lambda: True)
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(backend = "cuda")
    )
    monkeypatch.setattr(bk, "SdCppEngine", lambda binary: types.SimpleNamespace(
        binary = binary, is_available = lambda: True, version = lambda: "x"
    ))

    def _install(**kwargs):
        raise AssertionError(f"no reinstall may happen here (asked for {kwargs['accelerator']})")

    monkeypatch.setattr(sdmod, "install", _install)

    backend = bk.SdCppDiffusionBackend()
    assert backend._resolve_engine().binary == str(cli)
    assert cli.read_bytes() == b"cuda-build"


def test_a_serverless_deferred_install_still_lands_after_teardown(tmp_path, monkeypatch):
    """The serverless branch: only sd-cli is installed, so the deferral resolves the load to
    one-shot, and gating the post-teardown retry on mode == "server" skipped it for good. The
    archive carries the sd-cli this load generates with, so the install has to run anyway."""
    import core.inference.sd_cpp_backend as bk

    root = _managed_tree(tmp_path, monkeypatch, accelerator = "cpu")
    cli = root / "sd-bin" / "sd-cli"
    cli.write_bytes(b"cpu-build")
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: None)  # serverless install
    monkeypatch.setattr(bk, "find_sd_cpp_binary", lambda: str(cli))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    monkeypatch.setattr(bk, "_install_allowed", lambda: True)
    monkeypatch.setattr(
        bk, "resolve_diffusion_device_target", lambda: types.SimpleNamespace(backend = "cuda")
    )
    installs: list = []

    def _install(**kwargs):
        installs.append(kwargs)
        cli.write_bytes(b"cuda-build")
        sdmod._write_install_record(root, accelerator = kwargs["accelerator"], repo = "r", tag = "t")
        return cli

    monkeypatch.setattr(sdmod, "install", _install)

    backend = bk.SdCppDiffusionBackend()
    backend._state = types.SimpleNamespace(server = None)
    backend._active_generate_cancel = threading.Event()  # a one-shot sd-cli is still running
    mode, server_binary, _engine = backend._resolve_backend()
    assert mode == "oneshot" and server_binary is None
    assert installs == [] and backend._deferred_accelerator_install is True

    # After the teardown the tree is free, and the install lands even though the load resolved to
    # one-shot: sd-cli comes out of the same archive.
    backend._state = None
    backend._active_generate_cancel = None
    assert backend._upgrade_server_after_teardown(None) is None  # this archive ships no server
    assert [k["accelerator"] for k in installs] == ["cuda"], "the sd-cli still has to be upgraded"
    assert cli.read_bytes() == b"cuda-build"

    # And a matching tree is not reinstalled on the next deferred load.
    assert backend._upgrade_server_after_teardown(None) is None
    assert len(installs) == 1
