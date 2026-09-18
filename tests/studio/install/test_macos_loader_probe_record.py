# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for reusing a recorded macOS dyld probe.

The probe may be skipped only when the runtime digests and host profile still match.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from _pr10648_helpers import llama_host, load_studio_module  # noqa: E402

# Use an isolated module because tests monkeypatch its globals.
ILP = load_studio_module(
    "studio_install_llama_prebuilt_macos_loader_record", "install_llama_prebuilt.py"
)

RELEASE_TAG = "release-1"
UPSTREAM_TAG = "b9001"
PUBLISHED_REPO = "unslothai/llama.cpp"
INSTALL_KIND = "macos-arm64"


@pytest.fixture(autouse = True)
def _clear_full_check(monkeypatch):
    monkeypatch.delenv("UNSLOTH_PREBUILT_FULL_CHECK", raising = False)


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


def build_install(tmp_path: Path, host) -> Path:
    """Build an install whose marker passes the real fingerprint checks."""
    install_dir = tmp_path / "llama.cpp"
    runtime_dir = install_dir / "build" / "bin"
    runtime_dir.mkdir(parents = True)
    for name in ("llama-server", "llama-quantize", "llama-diffusion-gemma-visual-server"):
        for directory in (install_dir, runtime_dir):
            binary = directory / name
            binary.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
            binary.chmod(0o755)
    for dylib in ("libllama.0.dylib", "libggml.0.dylib", "libmtmd.0.dylib"):
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
    )
    return install_dir


def marker_of(install_dir: Path) -> dict:
    return json.loads((install_dir / "UNSLOTH_PREBUILT_INFO.json").read_text(encoding = "utf-8"))


def write_marker(install_dir: Path, marker: dict) -> None:
    (install_dir / "UNSLOTH_PREBUILT_INFO.json").write_text(
        json.dumps(marker, indent = 2) + "\n", encoding = "utf-8"
    )


def count_spawns(monkeypatch) -> list[int]:
    calls = [0]

    def _probe(*_args, **_kwargs):
        calls[0] += 1
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
