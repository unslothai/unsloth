# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for reusing a recorded macOS dyld probe.

The probe may be skipped only when the runtime digests and host profile still match.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from _pr10648_helpers import POSIX_ONLY, llama_host, load_studio_module  # noqa: E402

# Use an isolated module because tests monkeypatch its globals.
ILP = load_studio_module(
    "studio_install_llama_prebuilt_macos_loader_record", "install_llama_prebuilt.py"
)

RELEASE_TAG = "release-1"
UPSTREAM_TAG = "b9001"
PUBLISHED_REPO = "unslothai/llama.cpp"
INSTALL_KIND = "macos-arm64"


PRODUCT_VERSION = "15.5.1"
# Captured before the autouse fixture stubs it, for the tests that exercise the real one.
_REAL_MACOS_PRODUCT_VERSION = ILP.macos_product_version


@pytest.fixture(autouse = True)
def _clear_full_check(monkeypatch):
    monkeypatch.delenv("UNSLOTH_PREBUILT_FULL_CHECK", raising = False)
    # platform.mac_ver() is empty off macOS, so the probe record would never be written
    # and every test here would read as "no evidence". Pin the host's own answer instead.
    monkeypatch.setattr(ILP, "macos_product_version", lambda: PRODUCT_VERSION)


def macos_host(**overrides):
    return llama_host(
        ILP.HostInfo,
        system = "Darwin",
        machine = "arm64",
        macos_version = overrides.pop("macos_version", (15, 5)),
        **overrides,
    )


def choice_for(host) -> "ILP.AssetChoice":
    return ILP.AssetChoice(
        repo = PUBLISHED_REPO,
        tag = RELEASE_TAG,
        name = f"llama-{UPSTREAM_TAG}-bin-{INSTALL_KIND}.tar.gz",
        url = f"https://example.com/llama-{UPSTREAM_TAG}-bin-{INSTALL_KIND}.tar.gz",
        source_label = "published",
        install_kind = INSTALL_KIND,
        expected_sha256 = "a" * 64,
    )


def checksums_for(choice) -> "ILP.ApprovedReleaseChecksums":
    logical = ILP.source_archive_logical_name(UPSTREAM_TAG)
    return ILP.ApprovedReleaseChecksums(
        repo = PUBLISHED_REPO,
        release_tag = RELEASE_TAG,
        upstream_tag = UPSTREAM_TAG,
        source_commit = "deadbeef",
        artifacts = {
            logical: ILP.ApprovedArtifactHash(
                asset_name = logical,
                sha256 = "b" * 64,
                repo = "ggml-org/llama.cpp",
                kind = "upstream-source",
            ),
            choice.name: ILP.ApprovedArtifactHash(
                asset_name = choice.name,
                sha256 = choice.expected_sha256,
                repo = PUBLISHED_REPO,
                kind = "prebuilt",
            ),
        },
    )


def build_install(
    tmp_path: Path,
    host,
    *,
    load_probe_passed: bool = True,
) -> Path:
    """Build an install whose marker passes the real fingerprint checks.

    *load_probe_passed* mirrors what the installer learned from its own preflight:
    True is the normal install, where dyld resolved every binary. False is the
    install whose probe timed out or could not spawn, which must not be remembered
    as a pass.
    """
    install_dir = tmp_path / "llama.cpp"
    runtime_dir = install_dir / "build" / "bin"
    runtime_dir.mkdir(parents = True)
    for name in ("llama-server", "llama-quantize", "llama-diffusion-gemma-visual-server"):
        for directory in (install_dir, runtime_dir):
            binary = directory / name
            binary.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
            binary.chmod(0o755)
    # The libraries a real macos-arm64 bundle carries, one per group in
    # runtime_payload_health_groups: libllama-common and the split ggml libraries ship
    # beside libllama, so a fixture with only three of them is thinner than any install
    # this code grades. Verified against llama-b11030-mix-5ff778e-bin-macos-arm64.tar.gz.
    for dylib in (
        "libllama-common.0.dylib",
        "libllama.0.dylib",
        "libggml.0.dylib",
        "libggml-base.0.dylib",
        "libggml-cpu.0.dylib",
        "libmtmd.0.dylib",
        # The entrypoints carry no entry code of their own since the upstream split, and
        # the release tag here does not parse as a build number, which is the strict side
        # of the same gate Windows has used all along.
        "libllama-server-impl.dylib",
        "libllama-quantize-impl.dylib",
    ):
        (runtime_dir / dylib).write_bytes(b"DYLIB")
    (install_dir / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python3\n", encoding = "utf-8")
    (install_dir / "gguf-py" / "gguf").mkdir(parents = True)

    choice = choice_for(host)
    ILP.write_prebuilt_metadata(
        install_dir,
        host = host,
        requested_tag = "latest",
        llama_tag = UPSTREAM_TAG,
        release_tag = RELEASE_TAG,
        choice = choice,
        approved_checksums = checksums_for(choice),
        prebuilt_fallback_used = False,
        backend_request = "auto",
        macos_load_probe_passed = load_probe_passed,
    )
    return install_dir


def marker_of(install_dir: Path) -> dict:
    return json.loads((install_dir / "UNSLOTH_PREBUILT_INFO.json").read_text(encoding = "utf-8"))


def write_marker(install_dir: Path, marker: dict) -> None:
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps(marker, indent = 2) + "\n", encoding = "utf-8"
    )


def count_spawns(monkeypatch) -> list[int]:
    """Stand in for a probe that RAN and loaded every binary.

    Reporting the load matters: the real probe fails open, so preflight treats an
    empty *loaded* as "could not run" and refuses to record a pass. A stub that only
    returned [] would silently test the no-evidence path instead.
    """
    calls = [0]

    def _probe(
        binaries,
        install_dir,
        host,
        *,
        loaded = None,
        **_kwargs,
    ):
        calls[0] += 1
        if loaded is not None:
            loaded.update(path.name for path in binaries)
        return []

    monkeypatch.setattr(ILP, "macos_dyld_load_issues", _probe)
    return calls


def matches_choice(install_dir: Path, host) -> bool:
    choice = choice_for(host)
    return ILP.existing_install_matches_choice(
        install_dir,
        host,
        llama_tag = UPSTREAM_TAG,
        release_tag = RELEASE_TAG,
        choice = choice,
        approved_checksums = checksums_for(choice),
    )


def fast_path(install_dir: Path, host) -> bool:
    route = ILP.BackendRoute(
        backend = None,
        host = host,
        published_repo = PUBLISHED_REPO,
        published_release_tag = RELEASE_TAG,
        persist_llama_backend = None,
        persist_rocm_gfx = None,
    )
    return ILP.existing_install_current_without_plan(
        install_dir,
        llama_tag = UPSTREAM_TAG,
        published_repo = PUBLISHED_REPO,
        published_release_tag = RELEASE_TAG,
        backend_request = "auto",
        force_cpu = False,
        route = route,
    )


def test_reuse_with_a_current_record_does_not_probe(tmp_path: Path, monkeypatch):
    host = macos_host()
    install_dir = build_install(tmp_path, host)
    calls = count_spawns(monkeypatch)

    assert matches_choice(install_dir, host) is True
    assert calls[0] == 0


def test_fast_path_with_a_current_record_does_not_probe(tmp_path: Path, monkeypatch):
    host = macos_host()
    install_dir = build_install(tmp_path, host)
    calls = count_spawns(monkeypatch)

    assert fast_path(install_dir, host) is True
    assert calls[0] == 0


def test_check_installed_with_a_current_record_does_not_probe(tmp_path: Path, monkeypatch):
    host = macos_host()
    install_dir = build_install(tmp_path, host)
    calls = count_spawns(monkeypatch)
    # The other spawn on this path: _existing_install_runs starts each binary itself.
    monkeypatch.setattr(ILP, "_binary_image_runs", lambda *a, **k: pytest.fail("started a binary"))

    assert ILP._existing_install_runs(install_dir, host) is True
    assert calls[0] == 0


def test_a_marker_without_a_byte_record_still_probes(tmp_path: Path, monkeypatch):
    host = macos_host()
    install_dir = build_install(tmp_path, host)
    marker = marker_of(install_dir)
    marker.pop("runtime_files")
    write_marker(install_dir, marker)
    calls = count_spawns(monkeypatch)

    assert matches_choice(install_dir, host) is True
    assert calls[0] == 1


def test_a_size_only_dylib_record_still_probes(tmp_path: Path, monkeypatch):
    host = macos_host()
    install_dir = build_install(tmp_path, host)
    marker = marker_of(install_dir)
    # A marker written before macOS hashed its payload tier.
    marker["runtime_files"]["build/bin/libggml.0.dylib"].pop("sha256")
    write_marker(install_dir, marker)
    calls = count_spawns(monkeypatch)

    assert matches_choice(install_dir, host) is True
    assert calls[0] == 1


@POSIX_ONLY
def test_a_same_size_dylib_rewrite_is_rejected(tmp_path: Path, monkeypatch):
    host = macos_host()
    install_dir = build_install(tmp_path, host)
    dylib = install_dir / "build" / "bin" / "libggml.0.dylib"
    before = dylib.stat().st_size
    dylib.write_bytes(b"ROTTE")
    assert dylib.stat().st_size == before
    calls = count_spawns(monkeypatch)

    assert matches_choice(install_dir, host) is False
    assert fast_path(install_dir, host) is False
    # Rejected on the record, before anything was started.
    assert calls[0] == 0
    # _existing_install_runs answers a different question, so it only loses the skip.
    assert ILP._existing_install_runs(install_dir, host) is True
    assert calls[0] == 1


def test_a_marker_without_a_host_profile_still_probes(tmp_path: Path, monkeypatch):
    host = macos_host()
    install_dir = build_install(tmp_path, host)
    marker = marker_of(install_dir)
    marker.pop("host_profile")
    write_marker(install_dir, marker)
    calls = count_spawns(monkeypatch)

    assert matches_choice(install_dir, host) is True
    assert calls[0] == 1


def test_a_macos_upgrade_probes_again(tmp_path: Path, monkeypatch):
    install_dir = build_install(tmp_path, macos_host(macos_version = (15, 5)))
    upgraded = macos_host(macos_version = (26, 0))
    calls = count_spawns(monkeypatch)
    # The minos read is the other half of the preflight and reads real Mach-O headers.
    monkeypatch.setattr(ILP, "macos_binary_minos_issues", lambda *a, **k: [])

    assert matches_choice(install_dir, upgraded) is True
    assert calls[0] == 1


def test_an_unknown_macos_version_still_probes(tmp_path: Path, monkeypatch):
    host = macos_host(macos_version = None)
    install_dir = build_install(tmp_path, host)
    calls = count_spawns(monkeypatch)
    monkeypatch.setattr(ILP, "_binary_image_runs", lambda *a, **k: True)

    assert matches_choice(install_dir, host) is True
    assert ILP._existing_install_runs(install_dir, host) is True
    assert calls[0] == 2


def test_full_check_forces_the_probe(tmp_path: Path, monkeypatch):
    host = macos_host()
    install_dir = build_install(tmp_path, host)
    monkeypatch.setenv("UNSLOTH_PREBUILT_FULL_CHECK", "1")
    calls = count_spawns(monkeypatch)

    assert matches_choice(install_dir, host) is True
    assert calls[0] == 1


def test_a_replaced_binary_is_still_rejected(tmp_path: Path, monkeypatch):
    host = macos_host()
    install_dir = build_install(tmp_path, host)
    (install_dir / "build" / "bin" / "llama-server").write_text(
        "#!/bin/sh\nexit 1\n# swapped\n", encoding = "utf-8"
    )
    calls = count_spawns(monkeypatch)

    assert matches_choice(install_dir, host) is False
    assert fast_path(install_dir, host) is False
    # Rejected on the record, before anything was started.
    assert calls[0] == 0


def test_a_bundle_that_cannot_load_is_rejected_when_it_is_probed(tmp_path: Path, monkeypatch):
    host = macos_host()
    install_dir = build_install(tmp_path, host)
    marker = marker_of(install_dir)
    marker.pop("runtime_files")
    write_marker(install_dir, marker)
    monkeypatch.setattr(
        ILP,
        "macos_dyld_load_issues",
        lambda *a, **k: ["llama-server: Library not loaded: /usr/lib/librdma.dylib"],
    )

    assert matches_choice(install_dir, host) is False


@POSIX_ONLY
def test_an_install_whose_probe_never_ran_is_not_recorded_as_a_pass(tmp_path: Path, monkeypatch):
    """The probe fails open, so "no issues" is not the same as "it loaded"."""
    host = macos_host()
    install_dir = build_install(tmp_path, host, load_probe_passed = False)

    assert ILP.MACOS_LOAD_PROBE_KEY not in marker_of(install_dir)
    calls = count_spawns(monkeypatch)
    # Every byte still matches; the missing evidence alone is what costs a probe.
    assert ILP._existing_install_runs(install_dir, host) is True
    assert calls[0] == 1


def test_a_timed_out_probe_reports_no_pass(tmp_path: Path, monkeypatch):
    host = macos_host()
    install_dir = build_install(tmp_path, host)
    runtime_dir = install_dir / "build" / "bin"
    binaries = [runtime_dir / "llama-server", runtime_dir / "llama-quantize"]

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd = "llama-server", timeout = 60)

    monkeypatch.setattr(ILP, "run_capture", _timeout)
    monkeypatch.setattr(ILP, "macos_binary_minos_issues", lambda *a, **k: [])

    # Still not a rejection: a loaded machine must not lose a healthy bundle.
    assert ILP.preflight_macos_installed_binaries(binaries, install_dir, host) is False


def test_a_probe_that_ran_reports_a_pass(tmp_path: Path, monkeypatch):
    host = macos_host()
    install_dir = build_install(tmp_path, host)
    runtime_dir = install_dir / "build" / "bin"
    binaries = [runtime_dir / "llama-server", runtime_dir / "llama-quantize"]

    monkeypatch.setattr(
        ILP,
        "run_capture",
        lambda *a, **k: types.SimpleNamespace(returncode = 0, stdout = "", stderr = ""),
    )
    monkeypatch.setattr(ILP, "macos_binary_minos_issues", lambda *a, **k: [])

    assert ILP.preflight_macos_installed_binaries(binaries, install_dir, host) is True


@POSIX_ONLY
def test_a_macos_patch_update_probes_again(tmp_path: Path, monkeypatch):
    """host_profile is (major, minor), so only the recorded product version sees this."""
    host = macos_host()
    install_dir = build_install(tmp_path, host)
    calls = count_spawns(monkeypatch)
    # 15.5.1 -> 15.5.2 leaves host_profile identical, but replaces the dyld shared cache.
    monkeypatch.setattr(ILP, "macos_product_version", lambda: "15.5.2")

    assert ILP._existing_install_runs(install_dir, host) is True
    assert calls[0] == 1


@POSIX_ONLY
def test_a_reuse_probe_that_passed_is_remembered(tmp_path: Path, monkeypatch):
    """Otherwise the skip is unreachable for every install not born with the record."""
    host = macos_host()
    install_dir = build_install(tmp_path, host, load_probe_passed = False)
    assert ILP.MACOS_LOAD_PROBE_KEY not in marker_of(install_dir)

    calls = count_spawns(monkeypatch)
    assert ILP._existing_install_runs(install_dir, host) is True
    assert calls[0] == 1  # this update pays for the probe
    assert marker_of(install_dir)[ILP.MACOS_LOAD_PROBE_KEY] == {
        "passed": True,
        "macos_product_version": PRODUCT_VERSION,
    }

    assert ILP._existing_install_runs(install_dir, host) is True
    assert calls[0] == 1  # the next one does not


@POSIX_ONLY
def test_a_legacy_size_only_marker_is_upgraded_by_a_passing_probe(tmp_path: Path, monkeypatch):
    """The shape every install written before this PR has: payload recorded, not hashed."""
    host = macos_host()
    install_dir = build_install(tmp_path, host, load_probe_passed = False)
    marker = marker_of(install_dir)
    for key, entry in marker["runtime_files"].items():
        if not key.endswith(("llama-server", "llama-quantize")):
            entry.pop("sha256", None)
    marker.pop(ILP.MACOS_LOAD_PROBE_KEY, None)
    write_marker(install_dir, marker)

    calls = count_spawns(monkeypatch)
    assert ILP._existing_install_runs(install_dir, host) is True
    assert calls[0] == 1
    upgraded = marker_of(install_dir)["runtime_files"]
    assert all(entry.get("sha256") for entry in upgraded.values())

    assert ILP._existing_install_runs(install_dir, host) is True
    assert calls[0] == 1


def test_the_reuse_fast_paths_do_not_record_a_pass(tmp_path: Path, monkeypatch):
    """They probe build/bin only, so they never reach a root wrapper and cannot vouch
    for a record that a later run uses to skip starting one."""
    host = macos_host()
    install_dir = build_install(tmp_path, host, load_probe_passed = False)
    calls = count_spawns(monkeypatch)

    assert matches_choice(install_dir, host) is True
    assert fast_path(install_dir, host) is True
    assert calls[0] == 2  # both probed
    assert ILP.MACOS_LOAD_PROBE_KEY not in marker_of(install_dir)


@POSIX_ONLY
def test_a_broken_root_wrapper_is_not_blessed_into_the_record(tmp_path: Path, monkeypatch):
    """The root entrypoint can be its own file, and it is started AFTER the dyld probe.

    Recording between the two would hash the broken wrapper into the digest record, and
    the next run would read that record as current and never start the wrapper again.
    """
    host = macos_host()
    install_dir = build_install(tmp_path, host, load_probe_passed = False)
    marker = marker_of(install_dir)
    for key, entry in marker["runtime_files"].items():  # the pre-change shape
        if not key.endswith(("llama-server", "llama-quantize")):
            entry.pop("sha256", None)
    write_marker(install_dir, marker)
    count_spawns(monkeypatch)
    broken = install_dir / "llama-server"

    def _runs(path, *_a, **_k):
        return Path(path) != broken

    monkeypatch.setattr(ILP, "_binary_image_runs", _runs)

    assert ILP._existing_install_runs(install_dir, host) is False
    marker = marker_of(install_dir)
    assert ILP.MACOS_LOAD_PROBE_KEY not in marker
    # And the legacy size-only record was not upgraded on the strength of it either.
    assert not all(entry.get("sha256") for entry in marker["runtime_files"].values())


def test_a_probe_that_could_not_run_is_not_remembered(tmp_path: Path, monkeypatch):
    """Only a real load is evidence, so a fail-open probe must not persist one."""
    host = macos_host()
    install_dir = build_install(tmp_path, host, load_probe_passed = False)
    monkeypatch.setattr(ILP, "macos_binary_minos_issues", lambda *a, **k: [])

    def _timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd = "llama-server", timeout = 60)

    monkeypatch.setattr(ILP, "run_capture", _timeout)
    monkeypatch.setattr(ILP, "_binary_image_runs", lambda *a, **k: True)

    assert ILP._existing_install_runs(install_dir, host) is True
    assert ILP.MACOS_LOAD_PROBE_KEY not in marker_of(install_dir)


@POSIX_ONLY
def test_a_rapid_security_response_probes_again(tmp_path: Path, monkeypatch):
    """An RSR leaves ProductVersion alone and moves ProductVersionExtra and the build."""
    host = macos_host()
    install_dir = build_install(tmp_path, host)
    calls = count_spawns(monkeypatch)
    monkeypatch.setattr(ILP, "macos_product_version", lambda: PRODUCT_VERSION + " (a) 22E772610a")

    assert ILP._existing_install_runs(install_dir, host) is True
    assert calls[0] == 1


def test_the_recorded_version_carries_the_rsr_and_build_fields(monkeypatch):
    """platform.mac_ver() alone cannot see an RSR, so sw_vers is what gets asked."""
    seen = []

    def _sw_vers(cmd, **_kwargs):
        seen.append(cmd)
        return types.SimpleNamespace(
            returncode = 0,
            stdout = (
                "ProductName:\tmacOS\n"
                "ProductVersion:\t13.3.1\n"
                "ProductVersionExtra:\t(a)\n"
                "BuildVersion:\t22E772610a\n"
            ),
            stderr = "",
        )

    monkeypatch.setattr(ILP.subprocess, "run", _sw_vers)
    assert _REAL_MACOS_PRODUCT_VERSION() == "13.3.1 (a) 22E772610a"
    # Bare: sw_vers documents single-dash options, so no flag is spelled at all.
    assert seen == [["/usr/bin/sw_vers"]]


def test_no_rsr_installed_still_yields_a_version(monkeypatch):
    """ProductVersionExtra is simply absent with no RSR, and predates macOS 13."""

    def _sw_vers(_cmd, **_kwargs):
        return types.SimpleNamespace(
            returncode = 0,
            stdout = "ProductName:\tmacOS\nProductVersion:\t15.5.1\nBuildVersion:\t24F74\n",
            stderr = "",
        )

    monkeypatch.setattr(ILP.subprocess, "run", _sw_vers)
    assert _REAL_MACOS_PRODUCT_VERSION() == "15.5.1 24F74"


def test_a_failing_sw_vers_falls_back_to_the_product_version(monkeypatch):
    """A fallback that still catches a point release beats recording nothing."""

    def _boom(_cmd, **_kwargs):
        raise OSError("sw_vers not found")

    monkeypatch.setattr(ILP.subprocess, "run", _boom)
    monkeypatch.setattr(ILP.platform, "mac_ver", lambda: ("15.5.1", ("", "", ""), "arm64"))
    assert _REAL_MACOS_PRODUCT_VERSION() == "15.5.1"


def test_the_same_macos_patch_level_still_skips(tmp_path: Path, monkeypatch):
    host = macos_host()
    install_dir = build_install(tmp_path, host)
    calls = count_spawns(monkeypatch)

    assert ILP._existing_install_runs(install_dir, host) is True
    assert calls[0] == 0
