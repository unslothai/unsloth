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
import builtins  # noqa: E402
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
    binary = studio_home / "stable-diffusion.cpp" / "build" / "bin" / "sd-cli"
    binary.parent.mkdir(parents = True)
    binary.write_bytes(b"x")
    assert eng.find_sd_cpp_binary() == str(binary)


def test_managed_root_is_under_the_studio_home_like_every_other_component(tmp_path, monkeypatch):
    """The sd.cpp tree installs *under* the Studio home, not beside it.

    llama.cpp (``default_managed_llama_dir``), whisper.cpp and node all place their tree at
    ``<studio home>/<component>``. sd.cpp used the home's *parent*, which put the tree outside the
    home the user chose."""
    from core.inference import sd_cpp_engine as eng

    studio_home = tmp_path / "sxs" / "studio_a"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    assert default_install_dir() == studio_home / "stable-diffusion.cpp"
    # The installer and the engine must agree, or one installs where the other never looks.
    assert eng.managed_install_root() == default_install_dir()


def test_a_relative_studio_home_does_not_put_the_install_in_the_working_directory(
    tmp_path, monkeypatch
):
    """A relative ``UNSLOTH_STUDIO_HOME`` used to collapse to the working directory.

    ``Path("home").parent`` is ``Path(".")``, so the managed root became ``./stable-diffusion.cpp``
    -- exactly the name ``git clone`` of the upstream project produces. A checkout sitting in the
    working directory then shadowed the managed install and the installer refused to run, because
    the target was a pre-existing non-empty directory without the ownership marker."""
    from core.inference import sd_cpp_engine as eng

    monkeypatch.chdir(tmp_path)
    (tmp_path / "stable-diffusion.cpp").mkdir()
    (tmp_path / "stable-diffusion.cpp" / "README.md").write_text("someone else's checkout")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", "relative_home")
    monkeypatch.delenv("STUDIO_HOME", raising = False)

    for root in (default_install_dir(), eng.managed_install_root()):
        assert root.is_absolute(), "a relative home must not leave the root relative"
        assert root == (tmp_path / "relative_home" / "stable-diffusion.cpp").resolve()
        assert root != (tmp_path / "stable-diffusion.cpp").resolve()


def test_the_legacy_sibling_tree_is_adopted_only_when_it_carries_the_marker(tmp_path, monkeypatch):
    """An install an older build really made is still found; a bare checkout is not.

    Back-compat is marker-gated on purpose: the old location is ``<home>/../stable-diffusion.cpp``,
    which for a relative home is the working directory, so an unmarked match there is far more
    likely to be someone's clone than a previous Studio install."""
    from core.inference import sd_cpp_engine as eng

    monkeypatch.delenv("SD_CLI_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_SD_CPP_PATH", raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    sibling = tmp_path / "stable-diffusion.cpp"
    binary = sibling / "build" / "bin" / "sd-cli"
    binary.parent.mkdir(parents = True)
    binary.write_bytes(b"x")

    # Unmarked: not ours, so neither discovered nor treated as replaceable.
    assert eng.legacy_sibling_install_root() is None
    assert eng.find_sd_cpp_binary() is None
    assert eng.is_managed_binary(str(binary)) is False

    # Marked by a previous install: found again, and still replaceable.
    (sibling / eng.OWNER_MARKER).touch()
    assert eng.legacy_sibling_install_root() == sibling
    assert eng.find_sd_cpp_binary() == str(binary)
    assert eng.is_managed_binary(str(binary)) is True


def test_the_legacy_default_studio_home_keeps_its_install_dir(tmp_path, monkeypatch):
    """``UNSLOTH_STUDIO_HOME=~/.unsloth/studio`` is the documented default, not a custom home:
    it must keep resolving to ``~/.unsloth/stable-diffusion.cpp`` rather than moving the tree to
    ``~/.unsloth/studio/stable-diffusion.cpp`` and orphaning every existing install."""
    from core.inference import sd_cpp_engine as eng

    fake_home = tmp_path / "home"
    (fake_home / ".unsloth" / "studio").mkdir(parents = True)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(fake_home / ".unsloth" / "studio"))
    monkeypatch.delenv("STUDIO_HOME", raising = False)
    expected = fake_home / ".unsloth" / "stable-diffusion.cpp"
    assert default_install_dir() == expected
    assert eng.managed_install_root() == expected


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


def test_the_shipped_pin_is_mirror_only(tmp_path, monkeypatch):
    """The default pin carries the H3 patch set, which only the mirror can build.

    If this ever goes back to a plain upstream tag (because upstream released the fixes and
    patches/ was emptied), the two tests below stop describing the shipped default, so fail here
    loudly rather than letting them quietly assert nothing."""
    assert sdmod.is_mirror_only_tag(DEFAULT_TAG)
    assert not sdmod.is_mirror_only_tag("master-813-bfbef5b")
    assert not sdmod.is_mirror_only_tag("")
    assert not sdmod.is_mirror_only_tag(None)


def test_a_mirror_only_pin_is_never_requested_upstream(tmp_path, monkeypatch):
    """Upstream cannot have a -u<id> tag by construction, so asking for it is a guaranteed 404.

    The mirror-only pin is instead TRANSLATED back to the upstream release it was built from, so
    a host the mirror does not build keeps a pinned install rather than silently degrading to
    upstream latest. Asking upstream for the literal -u<id> string is still wrong, and that is
    what this pins.

    The mirror must NOT serve here: when it does, the very first attempt succeeds and the upstream
    attempts are never reached, so the assertion would hold no matter what the ordering says."""
    zb = _zip_with_sd_cli()
    _stub_two_repos(
        monkeypatch,
        mirror_serves = False,
        upstream_serves = True,
        zip_bytes = zb,
        digest = "sha256:" + hashlib.sha256(zb).hexdigest(),
    )
    asked = []
    real = sdmod._fetch_release
    monkeypatch.setattr(
        sdmod,
        "_fetch_release",
        lambda tag = None, **kw: (asked.append((kw.get("repo"), tag)), real(tag, **kw))[1],
    )
    install(install_dir = tmp_path)
    # Never the literal -u<id> string, which upstream cannot have.
    assert (sdmod.UPSTREAM_FALLBACK_REPO, DEFAULT_TAG) not in asked
    # It asks upstream for the release the mirror built on top of instead, so the pin survives
    # translation rather than being dropped.
    upstream_pin = sdmod.upstream_tag_for(DEFAULT_TAG)
    assert upstream_pin != DEFAULT_TAG
    assert (sdmod.UPSTREAM_FALLBACK_REPO, upstream_pin) in asked
    # And because that pinned attempt succeeds, it never has to settle for upstream latest --
    # which is the whole point of translating rather than skipping.
    assert (sdmod.UPSTREAM_FALLBACK_REPO, None) not in asked


def test_falling_back_off_a_mirror_only_pin_warns_about_h3(tmp_path, monkeypatch, capsys):
    """The generic fallback line is not enough here.

    Every other model still works on an unpatched upstream build, so falling back beats having no
    native engine at all. H3 does not: it aborts on the default cfg-scale and on --vae-on-cpu, and
    a blanket --type silently renders a broken video rather than failing. A user who sees only
    "falling back to leejet" has no way to connect that to the H3 output they get."""
    zb = _zip_with_sd_cli()
    _stub_two_repos(
        monkeypatch,
        mirror_serves = False,
        upstream_serves = True,
        zip_bytes = zb,
        digest = "sha256:" + hashlib.sha256(zb).hexdigest(),
    )
    install(install_dir = tmp_path)
    err = capsys.readouterr().err
    assert "falling back to leejet/stable-diffusion.cpp" in err
    assert "lacks the MiniMax-H3 fixes" in err


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


def test_a_legacy_sibling_install_is_read_from_its_own_root(tmp_path, monkeypatch):
    """The accelerator record belongs to the tree the binary is in.

    A tree an older build installed BESIDE the Studio home is still found by the finder, but the
    current managed root is now under the home and holds nothing. Reading the record from there
    reports the install as unrecorded, and unrecorded reads as a mismatch for a GPU target: the
    matching CUDA bundle already on disk would be downloaded again on every load."""
    import core.inference.sd_cpp_backend as bk

    home = tmp_path / "sd-home" / "studio"
    home.mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    legacy = tmp_path / "sd-home" / "stable-diffusion.cpp"  # where the old build put it
    (legacy / "sd-bin").mkdir(parents = True)
    (legacy / ".unsloth-studio-owned").touch()
    sdmod._write_install_record(legacy, accelerator = "cuda", repo = "r", tag = "t")
    sdmod._INSTALLED_ACCELERATOR_MEMO.clear()  # the record must be read off disk, not memoised

    server = legacy / "sd-bin" / "sd-server"
    server.write_bytes(b"cuda-build")
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(server))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    installs: list = []
    monkeypatch.setattr(sdmod, "install", lambda **kw: installs.append(kw))

    assert bk.ensure_sd_server_binary(accelerator = "cuda") == str(server)
    assert (
        installs == []
    ), "the CUDA build recorded in the legacy root is already what was asked for"


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
    monkeypatch.setattr(
        bk,
        "SdCppEngine",
        lambda binary: types.SimpleNamespace(
            binary = binary, is_available = lambda: True, version = lambda: "x"
        ),
    )

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


def test_a_server_still_starting_also_holds_the_tree(tmp_path, monkeypatch):
    """The startup window: between spawning sd-server and committing it to _state, the load has
    published only _pending_server. A second /images/load asking for a different accelerator would
    read the tree as idle and extract over the executable that is starting -- and start() blocks
    for as long as the checkpoint takes to load, so the window is minutes, not milliseconds."""
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

    starting = bk.SdCppDiffusionBackend()
    starting._state = None  # not committed yet
    starting._pending_server = object()
    monkeypatch.setattr(bk, "_sd_cpp_backend", starting)

    assert bk.ensure_sd_server_binary(accelerator = "cuda") == str(server)
    assert installs == [], "the starting server's file may not be overwritten"
    assert server.read_bytes() == b"cpu-build"

    # Once it has committed and been torn down, the upgrade lands.
    starting._pending_server = None
    assert bk.ensure_sd_server_binary(accelerator = "cuda") == str(server)
    assert [k["accelerator"] for k in installs] == ["cuda"]


def test_a_serverless_install_is_not_replaced_under_a_running_cli(tmp_path, monkeypatch):
    """The gap a mismatch-only guard leaves: on a legacy install with no sd-server,
    find_sd_server_binary returns None, so _accelerator_changed is never consulted and the install
    runs unconditionally. The router calls this directly with installs enabled, so a one-shot
    sd-cli mid-generation would have the tree extracted over it."""
    import core.inference.sd_cpp_backend as bk

    root = _managed_tree(tmp_path, monkeypatch, accelerator = "cpu")
    cli = root / "sd-bin" / "sd-cli"
    cli.write_bytes(b"cpu-build")
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: None)  # serverless install
    monkeypatch.setattr(bk, "find_sd_cpp_binary", lambda: str(cli))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    installs: list = []

    def _install(**kwargs):
        installs.append(kwargs)
        cli.write_bytes(b"cuda-build")
        sdmod._write_install_record(root, accelerator = kwargs["accelerator"], repo = "r", tag = "t")
        return cli

    monkeypatch.setattr(sdmod, "install", _install)

    generating = bk.SdCppDiffusionBackend()
    generating._active_generate_cancel = threading.Event()  # a one-shot sd-cli is running
    monkeypatch.setattr(bk, "_sd_cpp_backend", generating)

    assert bk.ensure_sd_server_binary(accelerator = "cuda") is None
    assert bk.ensure_sd_cpp_binary(accelerator = "cuda") == str(cli)
    assert installs == [], "no extraction over a running sd-cli"
    assert cli.read_bytes() == b"cpu-build"

    # With nothing running the install goes ahead, so a first install is never blocked.
    monkeypatch.setattr(bk, "_sd_cpp_backend", None)
    bk.ensure_sd_server_binary(accelerator = "cuda")
    assert [k["accelerator"] for k in installs] == ["cuda"]


def test_the_tree_stays_in_use_until_a_stopping_server_is_gone(tmp_path, monkeypatch):
    """unload() clears _state and _pending_server under the lock and stops OUTSIDE it, because
    terminate can take seconds. In that window both fields say idle while the process is still
    running its own executable, so a router call asking for another accelerator could extract over
    it. The stop is counted, and the count keeps the tree marked in use."""
    import core.inference.sd_cpp_backend as bk

    backend = bk.SdCppDiffusionBackend()
    seen: list = []

    class _SlowServer:
        def stop(self):
            # What a concurrent router call would see while this process is still going down.
            seen.append(bk._tree_in_use(backend))

    backend._stop_server(_SlowServer())
    assert seen == [True], "the tree must read busy until stop() returns"
    assert bk._tree_in_use(backend) is False
    assert backend._stopping_servers == 0

    # A stop that raises must neither propagate nor leak the count.
    class _BadServer:
        def stop(self):
            raise RuntimeError("terminate failed")

    backend._stop_server(_BadServer())
    assert backend._stopping_servers == 0 and bk._tree_in_use(backend) is False


def test_a_failed_server_upgrade_is_not_retried_by_the_cli_probe(tmp_path, monkeypatch):
    """The router probes the server first and the CLI immediately after. With only a usable CPU
    sd-cli in the tree, the failed server upgrade left no record (fallback was None), so the CLI
    probe resolved and downloaded the very same bundle a second time inside one selection."""
    import core.inference.sd_cpp_backend as bk

    root = _managed_tree(tmp_path, monkeypatch, accelerator = "cpu")
    cli = root / "sd-bin" / "sd-cli"
    cli.write_bytes(b"cpu-build")
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: None)
    monkeypatch.setattr(bk, "find_sd_cpp_binary", lambda: str(cli))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    monkeypatch.setattr(bk, "_sd_cpp_backend", None)
    attempts: list = []

    def _boom(**kwargs):
        attempts.append(kwargs["accelerator"])
        raise RuntimeError("no cuda asset for this host")

    monkeypatch.setattr(sdmod, "install", _boom)

    assert bk.ensure_sd_server_binary(accelerator = "cuda") is None
    assert bk.ensure_sd_cpp_binary(accelerator = "cuda") == str(cli)
    assert attempts == ["cuda"], "the same bundle must not be resolved twice in one selection"


def test_an_archive_without_a_server_drops_the_previous_one(tmp_path, monkeypatch):
    """An upgrade whose archive ships only sd-cli left the OLD accelerator's sd-server in place,
    while the record labelled the whole tree as the new accelerator: the backend then rediscovered
    that stale server, trusted the record, and ran a CUDA request on the old CPU build forever."""
    target = tmp_path / "sd"
    (target / "sd-bin").mkdir(parents = True)
    stale = target / "sd-bin" / ("sd-server.exe" if sys.platform == "win32" else "sd-server")
    stale.write_bytes(b"old-cpu-server")

    cli_name = "sd-cli.exe" if sys.platform == "win32" else "sd-cli"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"bin/{cli_name}", "new")
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        assert sdmod._archive_ships_sd_server(zf) is False

    sdmod._discard_stale_sd_server(target)
    assert not stale.exists()
    assert sdmod._locate_sd_server(target) is None


def test_an_archive_that_does_ship_a_server_keeps_it(tmp_path):
    """The removal must key on the archive's member list, not on what is on disk: a bundle that
    ships its own sd-server has just written it, and deleting that would break every upgrade."""
    server_name = "sd-server.exe" if sys.platform == "win32" else "sd-server"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"bin/{server_name}", "new")
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        assert sdmod._archive_ships_sd_server(zf) is True


def test_the_stop_is_reserved_before_the_server_is_unpublished(tmp_path, monkeypatch):
    """The count has to be claimed in the SAME lock block that clears _state/_pending_server. Doing
    it inside the stop leaves a gap where the resident, pending, stopping and generating fields are
    all empty while the process is still executing out of the managed tree."""
    import core.inference.sd_cpp_backend as bk

    backend = bk.SdCppDiffusionBackend()
    seen: list = []

    class _Server:
        def stop(self):
            seen.append(bk._tree_in_use(backend))

        def is_alive(self):
            return True

    server = _Server()
    backend._state = types.SimpleNamespace(server = server, mode = "server")
    monkeypatch.setattr(backend, "status", lambda: {"loaded": False})

    backend.unload()
    assert seen == [True], "the tree must still read busy while the old server is going down"
    assert backend._stopping_servers == 0 and bk._tree_in_use(backend) is False


def test_a_failed_start_does_not_block_its_own_cli_fallback(tmp_path, monkeypatch):
    """The fallback resolves the one-shot engine after stopping the server it just started. With
    that stopped server still in _pending_server the tree reads busy, so the lazy sd-cli install
    the fallback depends on was disabled and the load re-raised the server error instead."""
    import core.inference.sd_cpp_backend as bk

    backend = bk.SdCppDiffusionBackend()
    server = object()
    backend._pending_server = server
    assert bk._tree_in_use(backend) is True
    # What the fallback now does before it resolves the engine.
    with backend._lock:
        if backend._pending_server is server:
            backend._pending_server = None
    assert bk._tree_in_use(backend) is False


def test_an_unwritable_record_does_not_cost_the_install_or_repeat_it(tmp_path, monkeypatch):
    """A GPU bundle that extracts fine but cannot write its record used to read as unrecorded
    forever, and unrecorded is a mismatch for a GPU target: every later selection re-downloaded the
    same bundle. The install still succeeds, and the accelerator is remembered for this process."""
    root = tmp_path / "sd"
    root.mkdir()
    (root / sdmod.INSTALL_RECORD).mkdir()  # a directory where the record file goes: open() fails
    sdmod._INSTALLED_ACCELERATOR_MEMO.clear()

    sdmod._write_install_record(root, accelerator = "cuda", repo = "r", tag = "t")
    assert sdmod.read_install_record(root) == {}  # nothing on disk, as expected
    assert sdmod.installed_accelerator(root) == "cuda"  # but not "unknown"


def test_a_stale_unwritable_record_does_not_outrank_what_was_just_installed(tmp_path, monkeypatch):
    """The nastier shape of the same failure: the old record is READABLE but cannot be replaced, so
    it keeps answering "cpu" after a successful cuda install and every later selection downloads
    the bundle again. What this process installed is strictly newer than what is on disk."""
    root = tmp_path / "sd"
    root.mkdir()
    with open(root / sdmod.INSTALL_RECORD, "w", encoding = "utf-8") as f:
        json.dump({"accelerator": "cpu", "repo": "r", "tag": "old"}, f)
    sdmod._INSTALLED_ACCELERATOR_MEMO.clear()

    real_open = builtins.open

    def _readonly_record(
        file,
        mode = "r",
        *a,
        **k,
    ):
        if str(file).endswith(sdmod.INSTALL_RECORD) and "w" in mode:
            raise OSError("permission denied")
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", _readonly_record)
    sdmod._write_install_record(root, accelerator = "cuda", repo = "r", tag = "t")
    monkeypatch.setattr(builtins, "open", real_open)

    # The stale file is still there and still says cpu ...
    assert sdmod.read_install_record(root)["accelerator"] == "cpu"
    # ... but the accelerator this tree actually holds is the one just installed.
    assert sdmod.installed_accelerator(root) == "cuda"


def test_a_stale_server_that_cannot_be_removed_fails_the_install(tmp_path, monkeypatch):
    """A leftover server is still RUNNABLE, so no downstream probe repairs it, and a record naming
    the new accelerator would make _accelerator_changed trust it forever. Withhold both."""
    root = tmp_path / "sd"
    root.mkdir()
    stale = root / ("sd-server.exe" if sys.platform == "win32" else "sd-server")
    stale.write_bytes(b"old-cpu-server")

    def _no_unlink(self, *a, **k):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "unlink", _no_unlink)
    with pytest.raises(RuntimeError) as exc:
        sdmod._discard_stale_sd_server(root)
    assert "superseded sd-server" in str(exc.value)


def test_the_legacy_lookup_uses_the_lexical_parent_of_a_symlinked_home(tmp_path, monkeypatch):
    """The old code took Path(home).parent, which does not resolve symlinks. For a home that IS a
    link, the tree an older build created sits next to the LINK, so resolving first looked in the
    wrong place: a needless re-download, and the old install left orphaned from uninstall too."""
    import core.inference.sd_cpp_engine as eng

    target = tmp_path / "elsewhere" / "studio"
    target.mkdir(parents = True)
    home = tmp_path / "studio-home"
    home.symlink_to(target, target_is_directory = True)
    legacy = tmp_path / "stable-diffusion.cpp"  # beside the LINK, where the old build put it
    legacy.mkdir()
    (legacy / ".unsloth-studio-owned").touch()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))

    # Resolving first lands in <tmp>/elsewhere, which holds nothing.
    assert not (tmp_path / "elsewhere" / "stable-diffusion.cpp").exists()
    assert eng.legacy_sibling_install_root() == legacy


def test_a_serverless_install_does_not_fall_back_to_the_legacy_server(tmp_path, monkeypatch):
    """The finder also probes the tree beside the Studio home, so when the bundle just installed
    ships no sd-server the hit can be the legacy one, built for another accelerator. Returning it
    ran a forced CUDA load on the old CPU build instead of the one-shot CLI."""
    import core.inference.sd_cpp_backend as bk

    home = tmp_path / "sd-home" / "studio"
    home.mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    legacy = tmp_path / "sd-home" / "stable-diffusion.cpp"
    (legacy / "sd-bin").mkdir(parents = True)
    (legacy / ".unsloth-studio-owned").touch()
    sdmod._write_install_record(legacy, accelerator = "cpu", repo = "r", tag = "t")
    sdmod._INSTALLED_ACCELERATOR_MEMO.clear()
    old_server = legacy / "sd-bin" / "sd-server"
    old_server.write_bytes(b"cpu-build")

    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(old_server))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    # The install "succeeds" but ships only the CLI, so the finder still sees the legacy server.
    monkeypatch.setattr(sdmod, "install", lambda **_kw: None)

    assert bk.ensure_sd_server_binary(accelerator = "cuda") is None


def test_a_serverless_install_is_not_downloaded_again_on_every_later_load(tmp_path, monkeypatch):
    """Rejecting the legacy server is only half the answer. The tree beside the Studio home keeps
    that mismatched server, so the next load found it again, judged the accelerator changed and
    reinstalled the bundle already sitting in the current root -- once per model load, forever.
    The completed matching install in that root is the authoritative one: serverless, not stale."""
    import core.inference.sd_cpp_backend as bk

    home = tmp_path / "sd-home" / "studio"
    home.mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    legacy = tmp_path / "sd-home" / "stable-diffusion.cpp"
    (legacy / "sd-bin").mkdir(parents = True)
    (legacy / ".unsloth-studio-owned").touch()
    sdmod._write_install_record(legacy, accelerator = "cpu", repo = "r", tag = "t")
    old_server = legacy / "sd-bin" / "sd-server"
    old_server.write_bytes(b"cpu-build")
    # The current root already holds the cuda bundle, and that bundle shipped no sd-server.
    current = home / "stable-diffusion.cpp"
    current.mkdir()
    (current / ".unsloth-studio-owned").touch()
    sdmod._write_install_record(current, accelerator = "cuda", repo = "r", tag = "t", ships_server = False)
    sdmod._INSTALLED_ACCELERATOR_MEMO.clear()

    installs: list[dict] = []
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(old_server))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    monkeypatch.setattr(sdmod, "install", lambda **kw: installs.append(kw))

    assert bk.ensure_sd_server_binary(accelerator = "cuda") is None
    assert installs == []  # the bundle is already here; nothing to download
    # ... and with installs switched off the answer is the same, not the wrong-build server.
    assert bk.ensure_sd_server_binary(accelerator = "cuda", allow_install = False) is None
    # ... and while the managed tree is busy, where _accelerator_changed reports "unchanged"
    # because an install would overwrite a running binary.
    monkeypatch.setattr(bk, "_managed_tree_in_use", lambda: True)
    assert bk.ensure_sd_server_binary(accelerator = "cuda") is None
    assert installs == []


def test_a_matching_legacy_server_is_still_preferred_over_the_one_shot_cli(tmp_path, monkeypatch):
    """The serverless guard must fire only on a MISMATCHED legacy server. One built for the
    accelerator being asked for is a working resident server, and dropping to the one-shot CLI
    would re-run the model load on every single generation."""
    import core.inference.sd_cpp_backend as bk

    home = tmp_path / "sd-home" / "studio"
    home.mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    legacy = tmp_path / "sd-home" / "stable-diffusion.cpp"
    (legacy / "sd-bin").mkdir(parents = True)
    (legacy / ".unsloth-studio-owned").touch()
    sdmod._write_install_record(legacy, accelerator = "cpu", repo = "r", tag = "t")
    server = legacy / "sd-bin" / "sd-server"
    server.write_bytes(b"cpu-build")
    current = home / "stable-diffusion.cpp"
    current.mkdir()
    (current / ".unsloth-studio-owned").touch()
    sdmod._write_install_record(current, accelerator = "cpu", repo = "r", tag = "t", ships_server = False)
    sdmod._INSTALLED_ACCELERATOR_MEMO.clear()

    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(server))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())

    assert bk.ensure_sd_server_binary(accelerator = "cpu") == str(server)


def test_a_deleted_server_still_reinstalls_rather_than_reading_as_serverless(tmp_path, monkeypatch):
    """A missing sd-server is not proof the bundle never had one: the runnability repair unlinks a
    broken managed server precisely so the next load puts it back. Only a record that positively
    says the bundle shipped none may suppress that reinstall, so an install predating the field
    keeps repairing itself."""
    import core.inference.sd_cpp_backend as bk

    home = tmp_path / "sd-home" / "studio"
    home.mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    legacy = tmp_path / "sd-home" / "stable-diffusion.cpp"
    (legacy / "sd-bin").mkdir(parents = True)
    (legacy / ".unsloth-studio-owned").touch()
    sdmod._write_install_record(legacy, accelerator = "cpu", repo = "r", tag = "t")
    old_server = legacy / "sd-bin" / "sd-server"
    old_server.write_bytes(b"cpu-build")
    current = home / "stable-diffusion.cpp"
    current.mkdir()
    (current / ".unsloth-studio-owned").touch()
    # A record from before ships_server existed: the bundle's server capability is unknown.
    sdmod._write_install_record(current, accelerator = "cuda", repo = "r", tag = "t")
    sdmod._INSTALLED_ACCELERATOR_MEMO.clear()
    assert sdmod.installed_ships_server(current) is None

    installs: list[dict] = []
    monkeypatch.setattr(bk, "find_sd_server_binary", lambda: str(old_server))
    monkeypatch.setattr(bk, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(bk, "_failed_accelerator_upgrades", set())
    monkeypatch.setattr(sdmod, "install", lambda **kw: installs.append(kw))

    bk.ensure_sd_server_binary(accelerator = "cuda")
    assert [kw["accelerator"] for kw in installs] == ["cuda"]


def test_install_records_that_the_bundle_shipped_no_server(tmp_path, monkeypatch):
    """The capability comes off the archive member list during install(), so a later load reads it
    as fact. Without it, "no sd-server in the tree" is indistinguishable from one that was deleted
    and the reinstall that would repair it gets suppressed."""
    zb = _zip_with_sd_cli()  # sd-cli only
    _stub_release(monkeypatch, zip_bytes = zb, digest = "sha256:" + hashlib.sha256(zb).hexdigest())
    install(install_dir = tmp_path)
    assert sdmod.installed_ships_server(tmp_path) is False


def test_install_records_that_the_bundle_shipped_a_server(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("build/bin/sd-cli", b"#!/bin/sh\necho sd-cli\n")
        zf.writestr("build/bin/sd-server", b"#!/bin/sh\necho sd-server\n")
    zb = buf.getvalue()
    _stub_release(monkeypatch, zip_bytes = zb, digest = "sha256:" + hashlib.sha256(zb).hexdigest())
    install(install_dir = tmp_path)
    assert sdmod.installed_ships_server(tmp_path) is True


def test_the_install_record_remembers_whether_the_bundle_shipped_a_server(tmp_path):
    """Read off the archive member list at install time, so no later load has to guess."""
    root = tmp_path / "sd"
    root.mkdir()
    sdmod._write_install_record(root, accelerator = "cpu", repo = "r", tag = "t")
    assert sdmod.installed_ships_server(root) is None  # unrecorded stays unknown
    sdmod._write_install_record(root, accelerator = "cpu", repo = "r", tag = "t", ships_server = True)
    assert sdmod.installed_ships_server(root) is True
    sdmod._write_install_record(root, accelerator = "cpu", repo = "r", tag = "t", ships_server = False)
    assert sdmod.installed_ships_server(root) is False
    assert sdmod.installed_accelerator(root) == "cpu"
