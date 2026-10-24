# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The fast path must not keep a broken install; only a sha256 digest catches a same-size flip."""

import dataclasses
import errno
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Callable

import pytest

from _pr10648_helpers import (
    IS_ROOT,
    NEEDS_CHOWN,
    NOT_ROOT,
    POSIX_ONLY,
    WINDOWS_HOST,
    llama_host,
    whisper_host,
    whisper_install_is_intact,
    whisper_selection_fields,
)
from _pr10648_helpers import load_studio_module as _load

LLAMA = _load("studio_install_llama_prebuilt_pr10648_integrity", "install_llama_prebuilt.py")
CORE = _load("studio_prebuilt_core_pr10648_integrity", "prebuilt_core.py")
NODE = _load("studio_install_node_prebuilt_pr10648_integrity", "install_node_prebuilt.py")
WHISPER = _load("studio_install_whisper_prebuilt_pr10648_integrity", "install_whisper_prebuilt.py")

# Imported, not copied, so the fixture cannot drift from the real tree.
import test_keep_install_backcompat_9979 as KEEP  # noqa: E402

MARKER_NAME = "UNSLOTH_PREBUILT_INFO.json"


LINUX = llama_host(LLAMA.HostInfo)
_UPSTREAM = ("ggml-org/llama.cpp", "upstream-prebuilt")
_SOURCE = ("ggml-org/llama.cpp", "upstream-source")


def _asset_choice(**overrides):
    name = overrides.pop("name", "llama-b9001-bin-ubuntu-x64.tar.gz")
    defaults = dict(
        repo = "unslothai/llama.cpp",
        tag = "release-1",
        name = name,
        url = f"https://example.com/{name}",
        source_label = "upstream",
        install_kind = "linux-cpu",
        expected_sha256 = "a" * 64,
    )
    defaults.update(overrides)
    return LLAMA.AssetChoice(**defaults)


def _artifact(asset_name: str, sha256: str, origin: "tuple[str, str]"):
    repo, kind = origin
    return LLAMA.ApprovedArtifactHash(asset_name = asset_name, sha256 = sha256, repo = repo, kind = kind)


def _release_checksums(*assets: "tuple[str, str, tuple[str, str]]"):
    logical = LLAMA.source_archive_logical_name("b9001")
    artifacts = {logical: _artifact(logical, "b" * 64, _SOURCE)}
    for asset_name, sha256, origin in assets:
        artifacts[asset_name] = _artifact(asset_name, sha256, origin)
    return LLAMA.ApprovedReleaseChecksums(
        repo = "unslothai/llama.cpp",
        release_tag = "release-1",
        upstream_tag = "b9001",
        source_commit = "deadbeef",
        artifacts = artifacts,
    )


def _fill_payload(install_dir: Path, host) -> None:
    """Fills empty payload files with non-empty bytes; empty files cannot be truncated or bit-flipped."""
    runtime_dir = LLAMA.install_runtime_dir(install_dir, host)
    for path in sorted(runtime_dir.iterdir()):
        if path.is_file() and path.stat().st_size == 0:
            path.write_bytes(b"payload:" + path.name.encode("utf-8") + b":" + b"\xa5" * 48)


def _install(
    tmp_path: Path,
    monkeypatch,
    *,
    host = LINUX,
) -> Path:
    """Marker comes from real write_prebuilt_metadata: a hand-written one fails the fingerprint guard."""
    install_dir = KEEP.build_install(tmp_path, host = host, marker = None)
    _fill_payload(install_dir, host)
    choice = _asset_choice()
    checksums = _release_checksums((choice.name, choice.expected_sha256, _UPSTREAM))
    LLAMA.write_prebuilt_metadata(
        install_dir,
        host = host,
        requested_tag = "latest",
        llama_tag = "b9001",
        release_tag = "release-1",
        choice = choice,
        approved_checksums = checksums,
        prebuilt_fallback_used = False,
        backend_request = "auto",
    )
    monkeypatch.setattr(LLAMA, "detect_host", lambda **_k: host)
    # The one HEAD the precheck makes. Answered locally: this file never reaches the network.
    monkeypatch.setattr(LLAMA, "_download_host_latest_release_tag", lambda _repo: "release-1")
    for name in (
        "UNSLOTH_PREBUILT_FULL_CHECK",
        "UNSLOTH_LLAMA_DISABLE_DOWNLOAD_HOST_RESOLVE",
        "UNSLOTH_ROCM_GFX_ARCH",
        "UNSLOTH_ROCM_GFX_REMEMBERED",
    ):
        monkeypatch.delenv(name, raising = False)
    return install_dir


def _fast_path(install_dir: Path, **overrides) -> bool:
    kwargs = dict(
        llama_tag = "latest",
        published_repo = "unslothai/llama.cpp",
        published_release_tag = "",
        backend_request = "auto",
        force_cpu = False,
    )
    kwargs.update(overrides)
    return LLAMA.existing_install_current_without_plan(install_dir, **kwargs)


def _files_match(install_dir: Path, host = LINUX) -> bool:
    return LLAMA._runtime_files_match(install_dir, host, LLAMA.load_prebuilt_metadata(install_dir))


def _marker(install_dir: Path) -> dict:
    return json.loads((install_dir / MARKER_NAME).read_text(encoding = "utf-8"))


def _rewrite_marker(install_dir: Path, payload: dict) -> None:
    (install_dir / MARKER_NAME).write_text(json.dumps(payload, indent = 2) + "\n", encoding = "utf-8")


def _truncate_half(path: Path) -> None:
    data = path.read_bytes()
    assert len(data) > 2
    path.write_bytes(data[: len(data) // 2])


def _truncate_zero(path: Path) -> None:
    path.write_bytes(b"")


def _flip_one_byte(path: Path) -> None:
    """One bit, same length: the corruption a size check cannot see."""
    before = path.stat().st_size
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0x01
    path.write_bytes(bytes(data))
    assert path.stat().st_size == before, "the flip must preserve the size or it proves nothing"


def _delete(path: Path) -> None:
    path.unlink()


def _swap_for_another_binary(path: Path) -> None:
    """A different WORKING binary of a different size, as a botched manual repair leaves."""
    before = path.stat().st_size
    replacement = Path("/bin/sh")
    if not replacement.is_file():
        pytest.skip("no /bin/sh to stand in for a different working binary")
    path.unlink()
    shutil.copyfile(replacement, path)
    path.chmod(0o755)
    assert path.stat().st_size != before


_CORRUPTIONS = {
    "truncate_half": _truncate_half,
    "truncate_zero": _truncate_zero,
    "flip_one_byte": _flip_one_byte,
    "delete": _delete,
}

# Both copies of each binary: the root and build/bin layouts can rot independently.
HASHED_TIER = (
    "llama-server",
    "llama-quantize",
    "build/bin/llama-server",
    "build/bin/llama-quantize",
    "build/bin/llama-diffusion-gemma-visual-server",
)
# Allowlisted payload matched by lib*.so*: recorded size + mtime_ns, no digest.
SIZE_TIER = ("build/bin/libggml.so", "build/bin/libllama.so")


def test_the_healthy_install_is_accepted_by_the_fast_path(tmp_path, monkeypatch):
    """Each corruption test re-asserts this healthy baseline first, so a green result proves the damage."""
    install_dir = _install(tmp_path, monkeypatch)

    def boom(*_a, **_k):
        raise AssertionError("the precheck must not reach the GitHub API")

    monkeypatch.setattr(LLAMA, "fetch_release_bundle", boom, raising = False)
    monkeypatch.setattr(LLAMA, "resolve_release_tag", boom, raising = False)
    assert _fast_path(install_dir) is True
    assert _files_match(install_dir) is True
    recorded = _marker(install_dir)["runtime_files"]
    for relative in HASHED_TIER:
        assert recorded[relative].get("sha256"), f"{relative} must carry a digest"
    for relative in SIZE_TIER:
        assert "sha256" not in recorded[relative], f"{relative} is the size tier"


@pytest.mark.parametrize("relative", HASHED_TIER)
@pytest.mark.parametrize("corruption", sorted(_CORRUPTIONS))
def test_a_corrupted_recorded_binary_is_rejected(tmp_path, monkeypatch, relative, corruption):
    install_dir = _install(tmp_path, monkeypatch)
    assert _fast_path(install_dir) is True
    _CORRUPTIONS[corruption](install_dir / relative)
    assert _fast_path(install_dir) is False
    assert _files_match(install_dir) is False


@pytest.mark.parametrize("relative", HASHED_TIER)
def test_llama_a_same_size_byte_flip_is_caught_only_by_the_digest(tmp_path, monkeypatch, relative):
    """A same-size byte flip passes every structural check; only the recorded sha256 catches it."""
    install_dir = _install(tmp_path, monkeypatch)
    assert _fast_path(install_dir) is True
    recorded_size = _marker(install_dir)["runtime_files"][relative]["size"]

    _flip_one_byte(install_dir / relative)
    assert (install_dir / relative).stat().st_size == recorded_size
    assert _files_match(install_dir) is False
    assert _fast_path(install_dir) is False

    payload = _marker(install_dir)
    payload["runtime_files"][relative].pop("sha256")
    _rewrite_marker(install_dir, payload)
    assert _fast_path(install_dir) is True, "size alone cannot see a same-size flip"


@POSIX_ONLY
@pytest.mark.parametrize("relative", HASHED_TIER)
def test_a_recorded_binary_replaced_by_a_different_one_is_rejected(tmp_path, monkeypatch, relative):
    install_dir = _install(tmp_path, monkeypatch)
    assert _fast_path(install_dir) is True
    _swap_for_another_binary(install_dir / relative)
    assert _files_match(install_dir) is False
    assert _fast_path(install_dir) is False


@POSIX_ONLY
@pytest.mark.parametrize("relative", HASHED_TIER)
def test_a_recorded_binary_replaced_by_a_dangling_symlink_is_rejected(
    tmp_path, monkeypatch, relative
):
    """stat() follows the link, so this arrives as ENOENT rather than as a size mismatch;
    _runtime_files_match's OSError branch is what has to catch it."""
    install_dir = _install(tmp_path, monkeypatch)
    assert _fast_path(install_dir) is True
    target = install_dir / relative
    target.unlink()
    target.symlink_to(target.parent / "gone-with-the-install")
    assert target.is_symlink() and not target.exists()
    assert _files_match(install_dir) is False
    assert _fast_path(install_dir) is False


@POSIX_ONLY
@NOT_ROOT
@pytest.mark.parametrize("relative", HASHED_TIER)
def test_an_unreadable_recorded_binary_is_rejected(tmp_path, monkeypatch, relative):
    """chmod 000 leaves size and mtime intact, so only the hash attempt can fail -- and
    _runtime_files_match fails CLOSED on it, unlike the payload scans."""
    install_dir = _install(tmp_path, monkeypatch)
    assert _fast_path(install_dir) is True
    target = install_dir / relative
    target.chmod(0o000)
    try:
        assert target.stat().st_size == _marker(install_dir)["runtime_files"][relative]["size"]
        assert _files_match(install_dir) is False
        assert _fast_path(install_dir) is False
    finally:
        target.chmod(0o755)


@POSIX_ONLY
@NOT_ROOT
@pytest.mark.parametrize("relative", ("build/bin/llama-server", "build/bin/llama-quantize"))
def test_a_non_executable_runtime_binary_is_rejected(tmp_path, monkeypatch, relative):
    """The bytes are untouched, so the digests still match; os.access(X_OK) on the two
    runtime-dir binaries is the only thing standing between this and "already matches"."""
    install_dir = _install(tmp_path, monkeypatch)
    assert _fast_path(install_dir) is True
    (install_dir / relative).chmod(0o644)
    assert _files_match(install_dir) is True, "the record is about bytes, not mode"
    assert _fast_path(install_dir) is False


@POSIX_ONLY
@NOT_ROOT
@pytest.mark.parametrize(
    "relative,is_an_entrypoint",
    (
        ("llama-server", True),
        ("llama-quantize", True),
        ("build/bin/llama-diffusion-gemma-visual-server", False),
    ),
)
def test_the_execute_bit_is_checked_on_every_entrypoint(
    tmp_path, monkeypatch, relative, is_an_entrypoint
):
    """Which files losing their execute bit stop an install being kept, and by which check.

    Both halves ask _damaged_entrypoint now, so they answer together: it covers
    llama-{server,quantize} under build/bin AND the install root's copies, which
    _find_llama_server_binary reaches first. The shortcut used to check build/bin only,
    which let the desktop mark an install stale over a root copy and then have the update
    keep it unchanged; that loop is what this row pins.

    The visual server is not an entrypoint either check probes, and the blast radius there
    is bounded: llama_server_candidates keeps scanning the same layout and finds the
    executable build/bin copy. Its row is the control -- without it the fix could be
    "reject everything", which would send every install through a needless repair.
    """
    install_dir = _install(tmp_path, monkeypatch)
    assert _fast_path(install_dir) is True
    assert LLAMA._existing_install_runs(install_dir, LINUX) is True
    (install_dir / relative).chmod(0o644)
    assert _fast_path(install_dir) is not is_an_entrypoint
    assert LLAMA._existing_install_runs(install_dir, LINUX) is not is_an_entrypoint


@pytest.mark.parametrize("relative", SIZE_TIER)
@pytest.mark.parametrize("corruption", ("truncate_half", "truncate_zero", "delete"))
def test_a_truncated_payload_library_is_rejected(tmp_path, monkeypatch, relative, corruption):
    """The size tier's whole purpose: _runtime_payload_has globs for EXISTENCE, so before
    the record a half-written libggml.so -- what a full disk leaves -- passed the payload
    scan. A stat is enough to catch it."""
    install_dir = _install(tmp_path, monkeypatch)
    assert _fast_path(install_dir) is True
    _CORRUPTIONS[corruption](install_dir / relative)
    assert _files_match(install_dir) is False
    assert _fast_path(install_dir) is False


@pytest.mark.parametrize("relative", SIZE_TIER)
def test_a_same_size_payload_rewrite_is_not_detected(tmp_path, monkeypatch, relative):
    """Documents the size-tier edge: a same-size rewrite of a shared library is not detected, by design."""
    install_dir = _install(tmp_path, monkeypatch)
    assert _fast_path(install_dir) is True
    target = install_dir / relative
    target.write_bytes(b"\x00" * target.stat().st_size)
    assert _files_match(install_dir) is True
    assert _fast_path(install_dir) is True
    assert LLAMA._existing_install_runs(install_dir, LINUX) is True


def test_runtime_files_is_not_an_input_to_the_marker_fingerprint(tmp_path, monkeypatch):
    """runtime_files is outside the marker fingerprint; a rewritten record passes the consistency guard."""
    install_dir = _install(tmp_path, monkeypatch)
    marker = _marker(install_dir)
    before = LLAMA._marker_install_fingerprint(marker)
    assert before == marker["install_fingerprint"]

    marker["runtime_files"] = {}
    assert LLAMA._marker_install_fingerprint(marker) == before
    marker["runtime_files"] = {"llama-server": {"size": 1}}
    assert LLAMA._marker_install_fingerprint(marker) == before

    marker["release_tag"] = "release-2"
    assert LLAMA._marker_install_fingerprint(marker) != before


def test_dropping_the_digest_from_one_entry_downgrades_it_to_the_size_tier(tmp_path, monkeypatch):
    """An entry without sha256 is checked on size alone; the digest tier is just the key's presence."""
    install_dir = _install(tmp_path, monkeypatch)
    assert _fast_path(install_dir) is True

    payload = _marker(install_dir)
    entry = payload["runtime_files"]["llama-server"]
    assert entry.pop("sha256")
    assert entry["size"] > 0
    _rewrite_marker(install_dir, payload)
    assert _fast_path(install_dir) is True, "an entry without a digest is still a valid record"

    _flip_one_byte(install_dir / "llama-server")
    assert _files_match(install_dir) is True
    assert _fast_path(install_dir) is True

    _truncate_half(install_dir / "llama-server")
    assert _fast_path(install_dir) is False


def test_an_empty_runtime_files_record_fails_closed(tmp_path, monkeypatch):
    """(b) The record is the evidence that replaces starting the binaries, so no record
    is not proof of anything."""
    install_dir = _install(tmp_path, monkeypatch)
    assert _fast_path(install_dir) is True
    payload = _marker(install_dir)
    payload["runtime_files"] = {}
    _rewrite_marker(install_dir, payload)
    assert _files_match(install_dir) is False
    assert _fast_path(install_dir) is False


@pytest.mark.parametrize("record", (None, "", 0, [], "abc", ["size"], 17))
def test_a_runtime_files_record_that_is_not_a_dict_fails_closed(tmp_path, monkeypatch, record):
    install_dir = _install(tmp_path, monkeypatch)
    payload = _marker(install_dir)
    payload["runtime_files"] = record
    _rewrite_marker(install_dir, payload)
    assert _files_match(install_dir) is False
    assert _fast_path(install_dir) is False


def test_a_single_non_dict_entry_fails_the_whole_record_closed(tmp_path, monkeypatch):
    install_dir = _install(tmp_path, monkeypatch)
    payload = _marker(install_dir)
    payload["runtime_files"]["llama-server"] = "nonsense"
    _rewrite_marker(install_dir, payload)
    assert _files_match(install_dir) is False
    assert _fast_path(install_dir) is False


def test_a_marker_with_no_runtime_files_key_fails_closed(tmp_path, monkeypatch):
    """Every marker written before this PR is this shape: it takes the full path once,
    which re-records it."""
    install_dir = _install(tmp_path, monkeypatch)
    payload = _marker(install_dir)
    payload.pop("runtime_files")
    _rewrite_marker(install_dir, payload)
    assert _files_match(install_dir) is False
    assert _fast_path(install_dir) is False


@pytest.mark.parametrize("relative", ("llama-server", "build/bin/llama-server"))
def test_deleting_one_entry_leaves_only_that_file_unchecked(tmp_path, monkeypatch, relative):
    """Dropping one runtime_files entry leaves only that file unchecked; the other entries still reject."""
    install_dir = _install(tmp_path, monkeypatch)
    payload = _marker(install_dir)
    payload["runtime_files"].pop(relative)
    _rewrite_marker(install_dir, payload)
    assert _fast_path(install_dir) is True

    _flip_one_byte(install_dir / relative)
    assert _files_match(install_dir) is True
    assert _fast_path(install_dir) is True

    sibling = "llama-quantize" if relative == "llama-server" else "build/bin/llama-quantize"
    _flip_one_byte(install_dir / sibling)
    assert _files_match(install_dir) is False
    assert _fast_path(install_dir) is False


WHISPER_LINUX = whisper_host(WHISPER.HostInfo)
_GGML_TREE = "ggml-tree-aaaa"


def _whisper_selection(**overrides):
    # CORE.InstallSelection: this file loads its own prebuilt_core instance.
    return CORE.InstallSelection(**whisper_selection_fields(WHISPER, **overrides))


def _whisper_install(
    tmp_path: Path,
    monkeypatch,
    *,
    slim: bool = False,
) -> Path:
    """A whisper install tree plus a marker written by the REAL writer, so the
    fingerprint the keep path recomputes is genuinely self-consistent."""
    install_dir = tmp_path / "whisper.cpp"
    bin_dir = WHISPER.runtime_bin_dir(install_dir, WHISPER_LINUX)
    bin_dir.mkdir(parents = True)
    server = WHISPER.installed_server_path(install_dir, WHISPER_LINUX)
    server.write_bytes(b"#!/bin/sh\necho whisper\nexit 0\n")
    server.chmod(0o755)
    (bin_dir / "libwhisper.so.1").write_bytes(b"dummy-libwhisper")

    linked = ("libggml.so.0", "libggml-base.so.0")
    if slim:
        for name in linked:
            (bin_dir / name).write_bytes(b"ggml-" + name.encode("utf-8"))
    # The live llama marker a slim bundle is wired against; never the real one on this box.
    monkeypatch.setattr(WHISPER, "installed_llama_ggml_tree", lambda *_a, **_k: _GGML_TREE)

    selection = _whisper_selection(
        **(
            dict(
                install_kind = "slim",
                paired_llama_tag = "b9001",
                linked_from = str(tmp_path / "llama.cpp" / "build" / "bin"),
                linked_libraries = linked,
                runtime_wiring_version = WHISPER.SLIM_RUNTIME_WIRING_VERSION,
                linked_runtime_directories = (),
            )
            if slim
            else {}
        )
    )
    WHISPER.write_prebuilt_metadata(install_dir, selection)
    if slim:
        assert _whisper_marker(install_dir)["paired_llama_ggml_tree"] == _GGML_TREE
    return install_dir


def _whisper_marker(install_dir: Path) -> dict:
    return json.loads((install_dir / WHISPER.METADATA_FILENAME).read_text(encoding = "utf-8"))


def _whisper_keep(install_dir: Path) -> bool:
    return whisper_install_is_intact(WHISPER, install_dir, WHISPER_LINUX)


def test_whisper_a_healthy_install_is_kept(tmp_path, monkeypatch):
    install_dir = _whisper_install(tmp_path, monkeypatch)
    assert WHISPER.installed_tree_is_intact(install_dir, WHISPER_LINUX) is True
    assert _whisper_keep(install_dir) is True


def test_whisper_records_the_digest_of_the_server_it_installed(tmp_path, monkeypatch):
    """asset_sha256 covers only the archive; the marker separately records whisper-server's digest."""
    install_dir = _whisper_install(tmp_path, monkeypatch)
    marker = _whisper_marker(install_dir)
    assert marker["asset_sha256"] == "c" * 64
    server = WHISPER.installed_server_path(install_dir, WHISPER_LINUX)
    relative = server.relative_to(install_dir).as_posix()
    assert marker["runtime_files"][relative]["sha256"] == CORE.sha256_file(server)
    assert marker["runtime_files"][relative]["size"] == server.stat().st_size


def test_whisper_a_truncated_server_with_bytes_left_is_rejected(tmp_path, monkeypatch):
    """A truncated whisper-server passes every shape check; the recorded size and digest reject it."""
    install_dir = _whisper_install(tmp_path, monkeypatch)
    server = WHISPER.installed_server_path(install_dir, WHISPER_LINUX)
    assert _whisper_keep(install_dir) is True

    data = server.read_bytes()
    server.write_bytes(data[: len(data) // 2])
    assert server.stat().st_size > 0
    assert os.name == "nt" or os.access(server, os.X_OK)
    assert WHISPER.installed_tree_is_intact(install_dir, WHISPER_LINUX) is False
    assert _whisper_keep(install_dir) is False


def test_whisper_a_zero_byte_server_is_rejected(tmp_path, monkeypatch):
    """The one byte-level fact the whisper record can establish: size > 0."""
    install_dir = _whisper_install(tmp_path, monkeypatch)
    assert _whisper_keep(install_dir) is True
    WHISPER.installed_server_path(install_dir, WHISPER_LINUX).write_bytes(b"")
    assert WHISPER.installed_tree_is_intact(install_dir, WHISPER_LINUX) is False
    assert _whisper_keep(install_dir) is False


def test_whisper_a_missing_server_is_rejected(tmp_path, monkeypatch):
    install_dir = _whisper_install(tmp_path, monkeypatch)
    assert _whisper_keep(install_dir) is True
    WHISPER.installed_server_path(install_dir, WHISPER_LINUX).unlink()
    assert WHISPER.installed_tree_is_intact(install_dir, WHISPER_LINUX) is False
    assert _whisper_keep(install_dir) is False


@POSIX_ONLY
@NOT_ROOT
def test_whisper_a_non_executable_server_is_rejected(tmp_path, monkeypatch):
    install_dir = _whisper_install(tmp_path, monkeypatch)
    server = WHISPER.installed_server_path(install_dir, WHISPER_LINUX)
    assert _whisper_keep(install_dir) is True
    server.chmod(0o644)
    assert WHISPER.installed_tree_is_intact(install_dir, WHISPER_LINUX) is False
    assert _whisper_keep(install_dir) is False
    server.chmod(0o755)
    assert _whisper_keep(install_dir) is True


def test_whisper_a_slim_install_is_kept_only_while_its_paired_ggml_tree_stands(
    tmp_path, monkeypatch
):
    """A slim bundle ships no ggml of its own: it hardlinks llama's. So "intact" here is
    a statement about ANOTHER install, and a llama update that moved ggml retires it."""
    install_dir = _whisper_install(tmp_path, monkeypatch, slim = True)
    assert _whisper_keep(install_dir) is True

    monkeypatch.setattr(WHISPER, "installed_llama_ggml_tree", lambda *_a, **_k: "ggml-tree-bbbb")
    assert _whisper_keep(install_dir) is False
    assert WHISPER.installed_tree_is_intact(install_dir, WHISPER_LINUX) is True

    # A llama install that cannot say (predating ggml_tree) is not a licence to keep it.
    monkeypatch.setattr(WHISPER, "installed_llama_ggml_tree", lambda *_a, **_k: None)
    assert _whisper_keep(install_dir) is False


@pytest.mark.parametrize("missing", ("libggml.so.0", "libggml-base.so.0"))
def test_whisper_a_slim_install_missing_a_wired_library_is_rejected(tmp_path, monkeypatch, missing):
    install_dir = _whisper_install(tmp_path, monkeypatch, slim = True)
    assert _whisper_keep(install_dir) is True
    bin_dir = WHISPER.runtime_bin_dir(install_dir, WHISPER_LINUX)
    (bin_dir / missing).unlink()
    assert WHISPER.installed_tree_is_intact(install_dir, WHISPER_LINUX) is False
    assert _whisper_keep(install_dir) is False


def test_whisper_a_wired_library_truncated_to_one_byte_is_rejected(tmp_path, monkeypatch):
    """A ggml library truncated to a stub passes the by-name presence check; the recorded size
    catches it."""
    install_dir = _whisper_install(tmp_path, monkeypatch, slim = True)
    bin_dir = WHISPER.runtime_bin_dir(install_dir, WHISPER_LINUX)
    (bin_dir / "libggml.so.0").write_bytes(b"\x00")
    assert WHISPER.installed_tree_is_intact(install_dir, WHISPER_LINUX) is False
    assert _whisper_keep(install_dir) is False


@dataclasses.dataclass(frozen = True)
class _Writer:
    name: str
    module: Any
    filename: str
    rewrite: Callable[[Path, dict], Any]
    raises_on_failure: bool


WRITERS = (
    _Writer(
        name = "llama._write_marker",
        module = LLAMA,
        filename = MARKER_NAME,
        rewrite = lambda directory, payload: LLAMA._write_marker(directory / MARKER_NAME, payload),
        raises_on_failure = False,
    ),
    _Writer(
        name = "prebuilt_core.write_live_marker",
        module = CORE,
        filename = MARKER_NAME,
        rewrite = lambda directory, payload: CORE.write_live_marker(directory / MARKER_NAME, payload),
        raises_on_failure = True,
    ),
    _Writer(
        name = "node._write_metadata_payload",
        module = NODE,
        filename = NODE.METADATA_FILENAME,
        rewrite = lambda directory, payload: NODE._write_metadata_payload(directory, payload),
        raises_on_failure = True,
    ),
)
_WRITER_IDS = [writer.name for writer in WRITERS]

# release_tag and tag are shown in the Studio About tab via /api/system/hardware.
_LIVE_PAYLOAD = {
    "release_tag": "release-1",
    "tag": "b9001",
    "version": "24.9.0",
    "install_fingerprint": "ab" * 32,
    "nested": {"runtime_files": {"llama-server": {"size": 17}}},
}


def _live_marker(
    tmp_path: Path,
    writer: _Writer,
    *,
    mode: int = 0o644,
) -> Path:
    path = tmp_path / writer.filename
    path.write_text(
        json.dumps({"release_tag": "release-0", "tag": "b9000"}) + "\n", encoding = "utf-8"
    )
    path.chmod(mode)
    return path


def _temp_siblings(directory: Path) -> list:
    return sorted(p.name for p in directory.iterdir() if ".tmp-" in p.name)


@pytest.mark.parametrize("writer", WRITERS, ids = _WRITER_IDS)
def test_a_rewritten_marker_is_valid_json_and_keeps_the_keys_the_ui_reads(tmp_path, writer):
    _live_marker(tmp_path, writer)
    writer.rewrite(tmp_path, _LIVE_PAYLOAD)
    written = json.loads((tmp_path / writer.filename).read_text(encoding = "utf-8"))
    assert written == _LIVE_PAYLOAD
    assert written["release_tag"] == "release-1" and written["tag"] == "b9001"
    assert _temp_siblings(tmp_path) == []


@POSIX_ONLY
@pytest.mark.parametrize("mode", (0o600, 0o644, 0o664, 0o444))
@pytest.mark.parametrize("writer", WRITERS, ids = _WRITER_IDS)
def test_a_rewritten_marker_keeps_its_mode(tmp_path, writer, mode):
    """NamedTemporaryFile is 0600 and os.replace keeps the SOURCE file's mode, so without
    the restore a refresh silently makes a group-shared install's marker private. 0o444
    is the read-only case the temp-and-replace shape exists to support: a plain write
    would need the file writable, not the directory."""
    path = _live_marker(tmp_path, writer, mode = mode)
    writer.rewrite(tmp_path, _LIVE_PAYLOAD)
    assert stat.S_IMODE(path.stat().st_mode) == mode
    assert json.loads(path.read_text(encoding = "utf-8"))["tag"] == "b9001"
    assert _temp_siblings(tmp_path) == []


@POSIX_ONLY
@NEEDS_CHOWN
@NOT_ROOT
@pytest.mark.parametrize("writer", WRITERS, ids = _WRITER_IDS)
def test_a_group_shared_marker_keeps_its_group(tmp_path, writer):
    """A real chown, not a recorded call: the marker is put into a secondary group of
    this user and has to come back out of the rewrite in that group, because os.replace
    installs the TEMP file's ownership and NamedTemporaryFile's is the primary group."""
    groups = [gid for gid in os.getgroups() if gid != os.getegid()]
    if not groups:
        pytest.skip("this user is in no secondary group to share a marker with")
    path = _live_marker(tmp_path, writer)
    shared = groups[0]
    try:
        os.chown(path, -1, shared)
    except OSError as exc:  # pragma: no cover - depends on the mount
        pytest.skip(f"cannot regroup a file here ({exc})")
    assert path.stat().st_gid == shared

    writer.rewrite(tmp_path, _LIVE_PAYLOAD)
    assert path.stat().st_gid == shared
    assert json.loads(path.read_text(encoding = "utf-8"))["tag"] == "b9001"
    assert _temp_siblings(tmp_path) == []


@NEEDS_CHOWN
@pytest.mark.parametrize("writer", WRITERS, ids = _WRITER_IDS)
def test_every_marker_writer_asks_for_the_owner_then_falls_back_to_the_group(
    tmp_path, writer, monkeypatch
):
    """All three ask for owner AND group first, then group alone, and that uniformity is
    the point.

    Neither half is sufficient on its own. Owner+group alone is EPERM for a non-root member
    of a group-shared install -- chown is all-or-nothing -- so the group is silently lost,
    which is what e8d128d24 fixed in core and node. Group alone is wrong under root, which
    is exactly when the owner CAN be restored: it leaves the marker owned by root, and an
    0600 marker stops being readable by the user who owns the install. The two calls in
    this order give each caller the best it is permitted. This pins the call shape so the
    three writers cannot drift apart again.
    """
    path = _live_marker(tmp_path, writer)
    original = path.stat()
    calls: list = []

    def refusing_chown(target, uid, gid):
        calls.append((Path(target).name, uid, gid))
        if uid != -1:
            raise PermissionError("a non-root member may not give a file away")

    monkeypatch.setattr(writer.module.os, "chown", refusing_chown)
    writer.rewrite(tmp_path, _LIVE_PAYLOAD)
    monkeypatch.undo()

    assert [(uid, gid) for _, uid, gid in calls] == [
        (original.st_uid, original.st_gid),
        (-1, original.st_gid),
    ], calls
    assert all(
        ".tmp-" in name for name, _, _ in calls
    ), "ownership must be set on the temp file, before the swap"


@NEEDS_CHOWN
@pytest.mark.parametrize("writer", WRITERS, ids = _WRITER_IDS)
def test_a_permitted_writer_restores_the_owner_and_asks_no_further(tmp_path, writer, monkeypatch):
    """The root case: when the combined call is allowed, the fallback must not run, or the
    owner just restored would be left in place by luck rather than by intent."""
    path = _live_marker(tmp_path, writer)
    original = path.stat()
    calls: list = []

    monkeypatch.setattr(
        writer.module.os,
        "chown",
        lambda target, uid, gid: calls.append((Path(target).name, uid, gid)),
    )
    writer.rewrite(tmp_path, _LIVE_PAYLOAD)
    monkeypatch.undo()

    assert [(uid, gid) for _, uid, gid in calls] == [(original.st_uid, original.st_gid)], calls


@NEEDS_CHOWN
@NOT_ROOT
@pytest.mark.parametrize("writer", WRITERS, ids = _WRITER_IDS)
def test_a_marker_owned_by_another_user_still_keeps_its_group(tmp_path, writer, monkeypatch):
    """The case uid -1 exists for, on the install it matters for.

    A group-shared install whose marker is owned by the admin who ran setup, refreshed by
    a member of the group: chown(uid, gid) is EPERM outright there (you may not give a
    file away) while chown(-1, gid) succeeds, and since the failure is swallowed the
    difference is not an error but a marker that quietly changes group -- on an 0640
    marker, the mode restore beside it undone. The kernel rule is simulated, since a test
    cannot own a file as another user; the simulation refuses exactly what POSIX refuses.
    """
    path = _live_marker(tmp_path, writer, mode = 0o640)
    other_uid = os.geteuid() + 1
    shared_gid = os.getegid() + 1
    applied: list = []

    def kernel_chown(target, uid, gid):
        # POSIX: only root may change a file's owner; -1 means "leave it".
        if uid not in (-1, os.geteuid()):
            raise PermissionError(errno.EPERM, "Operation not permitted")
        applied.append((uid, gid))

    real_stat = os.stat
    monkeypatch.setattr(writer.module.os, "chown", kernel_chown)
    monkeypatch.setattr(
        writer.module.os,
        "stat",
        lambda *a, **k: _StatWithOwner(real_stat(*a, **k), other_uid, shared_gid),
    )
    writer.rewrite(tmp_path, _LIVE_PAYLOAD)
    # Narrow window: os.stat is the whole interpreter's, so it is restored immediately.
    monkeypatch.undo()
    assert json.loads(path.read_text(encoding = "utf-8"))["tag"] == "b9001"
    assert applied == [(-1, shared_gid)], "the group must survive a marker owned by someone else"


class _StatWithOwner:
    """An os.stat_result with st_uid/st_gid overridden, so a marker can stand in for one
    owned by another user without needing root to create it."""

    def __init__(self, base, uid: int, gid: int) -> None:
        self._base = base
        self.st_uid = uid
        self.st_gid = gid

    def __getattr__(self, name):
        return getattr(self._base, name)


@NEEDS_CHOWN
@pytest.mark.parametrize("writer", WRITERS, ids = _WRITER_IDS)
def test_a_chown_that_is_refused_does_not_abort_the_write(tmp_path, writer, monkeypatch):
    """Ownership is best effort; the refreshed marker is not. Declining to write because
    the group could not be restored would leave the field the refresh exists to record
    (a deliberate --force-cpu, a re-probed version) unrecorded."""
    path = _live_marker(tmp_path, writer, mode = 0o640)

    def refuse(*_a, **_k):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(writer.module.os, "chown", refuse)
    writer.rewrite(tmp_path, _LIVE_PAYLOAD)
    assert json.loads(path.read_text(encoding = "utf-8")) == _LIVE_PAYLOAD
    if not WINDOWS_HOST:
        assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert _temp_siblings(tmp_path) == []


@pytest.mark.parametrize("writer", WRITERS, ids = _WRITER_IDS)
def test_a_platform_with_no_os_chown_at_all_still_writes(tmp_path, writer, monkeypatch):
    """Windows: os.chown does not exist. Each writer catches AttributeError beside OSError
    for exactly this, and the rewrite has to complete anyway."""
    path = _live_marker(tmp_path, writer)
    monkeypatch.delattr(writer.module.os, "chown", raising = False)
    assert not hasattr(writer.module.os, "chown")
    writer.rewrite(tmp_path, _LIVE_PAYLOAD)
    assert json.loads(path.read_text(encoding = "utf-8")) == _LIVE_PAYLOAD
    assert _temp_siblings(tmp_path) == []


@pytest.mark.parametrize("writer", WRITERS, ids = _WRITER_IDS)
def test_a_replace_that_fails_leaves_the_previous_marker_whole(tmp_path, writer, monkeypatch):
    """Temp-and-replace: a failed swap must leave the previous marker byte-identical and no .tmp-* file."""
    path = _live_marker(tmp_path, writer)
    before = path.read_bytes()

    def boom(*_a, **_k):
        raise OSError(errno.EIO, "the disk went away mid-swap")

    monkeypatch.setattr(writer.module, "atomic_replace_from_tempfile", boom)
    if writer.raises_on_failure:
        with pytest.raises(OSError):
            writer.rewrite(tmp_path, _LIVE_PAYLOAD)
    else:
        assert writer.rewrite(tmp_path, _LIVE_PAYLOAD) is False

    assert path.read_bytes() == before
    assert json.loads(path.read_text(encoding = "utf-8"))["tag"] == "b9000"
    assert _temp_siblings(tmp_path) == []


@pytest.mark.parametrize("writer", WRITERS, ids = _WRITER_IDS)
def test_a_write_that_fails_before_the_swap_strands_no_temp_file(tmp_path, writer, monkeypatch):
    """The ENOSPC this shape is built to tolerate, raised from the write itself: the temp
    path is tracked outside the try precisely so it can still be removed."""
    path = _live_marker(tmp_path, writer)
    before = path.read_bytes()

    def boom(_fd):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(writer.module.os, "fsync", boom)
    if writer.raises_on_failure:
        with pytest.raises(OSError):
            writer.rewrite(tmp_path, _LIVE_PAYLOAD)
    else:
        assert writer.rewrite(tmp_path, _LIVE_PAYLOAD) is False
    monkeypatch.undo()

    assert path.read_bytes() == before
    assert _temp_siblings(tmp_path) == []


def test_the_llama_marker_survives_a_rewrite_and_the_fast_path_still_accepts_it(
    tmp_path, monkeypatch
):
    """A marker rewrite must keep the fields the precheck reads and the file's mode and group."""
    install_dir = _install(tmp_path, monkeypatch)
    marker_path = install_dir / MARKER_NAME
    marker_path.chmod(0o640)
    before = _marker(install_dir)
    assert _fast_path(install_dir) is True

    marker = dict(before)
    marker["force_cpu"] = False
    assert LLAMA._write_marker(marker_path, marker) is True

    after = _marker(install_dir)
    assert after == marker
    assert after["release_tag"] == before["release_tag"] == "release-1"
    assert after["tag"] == before["tag"] == "b9001"
    assert after["runtime_files"] == before["runtime_files"]
    assert LLAMA._marker_install_fingerprint(after) == after["install_fingerprint"]
    if not WINDOWS_HOST:
        assert stat.S_IMODE(marker_path.stat().st_mode) == 0o640
    assert _temp_siblings(install_dir) == []
    assert _fast_path(install_dir) is True


@POSIX_ONLY
@NOT_ROOT
@pytest.mark.parametrize("relative", ("llama-server", "llama-quantize"))
def test_a_damaged_root_entrypoint_is_not_kept_by_the_shortcut(tmp_path, monkeypatch, relative):
    """The update has to repair what the launch check calls broken, or the two loop.

    installed_runtime_health rejects a root llama-server that lost its execute bit, so the
    desktop marks the install stale and offers a repair. This shortcut returns before
    reinstalling anything, and it used to check X_OK under build/bin ONLY -- so the repair
    ran, changed nothing, and the next launch was stale again. Measured before the fix:
    launch health llama_runtime_binaries_missing, shortcut True, full re-validation False.

    _damaged_entrypoint owns the question for all three now, which is what its own docstring
    asks for.
    """
    install_dir = _install(tmp_path, monkeypatch)
    assert _fast_path(install_dir) is True, "the shortcut must accept a healthy install here"
    (install_dir / relative).chmod(0o644)
    assert LLAMA._existing_install_runs(install_dir, LINUX) is False
    assert _fast_path(install_dir) is False
