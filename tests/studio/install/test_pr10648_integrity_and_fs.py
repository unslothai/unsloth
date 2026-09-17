# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PR #10648: can the marker fast path keep a BROKEN prebuilt install?

``studio update`` used to re-validate a kept prebuilt by re-reading the release and
STARTING llama-server. The precheck replaces that with evidence recorded in the marker,
so the question every test here asks is the same one: take a healthy install, damage
exactly one thing, and see whether the fast path still says "already matches".

Three components, three different amounts of evidence, so three different answers:

  * llama (``_runtime_files_match``) records ``size`` for every allowlisted runtime file
    and ``size + sha256`` for the three binaries a reuse decision would otherwise run.
    Only the digest can see a corruption that preserves the byte count, which is why
    ``test_llama_a_same_size_byte_flip_is_caught_only_by_the_digest`` is the load-bearing test
    in this file.
  * whisper (``installed_tree_is_intact``) now records the same two tiers in the same
    ``runtime_files`` shape: ``size + sha256`` for ``whisper-server``, ``size`` for the
    ggml libraries a slim bundle hardlinks. It previously recorded NO payload digests at
    all -- it asked only that the server was a non-empty executable file and that a slim
    bundle's wiring was still there by name, so a truncation that left bytes behind was
    invisible to it. The three tests below that used to pin that weaker boundary now pin
    the digest one; a marker with no record (every install predating the key) is still
    kept and backfilled once, which ``test_pr10648_whisper_payload_digests.py`` owns.
  * node (``_file_record_matches``) records ``sha256`` and never compares it: ``size``
    and ``mtime_ns`` only, deliberately (see its docstring). Not exercised here beyond
    its marker writer; ``test_install_node_prebuilt_logic.py`` owns that decision.

Part 4 leaves integrity behind and tests the atomic marker rewriters the PR added
(``prebuilt_core.write_live_marker``, ``install_node_prebuilt._write_metadata_payload``,
``install_llama_prebuilt._write_marker``) as what they are: filesystem operations on a
file that is already in service, which must keep its mode and its group, must never
leave a ``.tmp-*`` sibling, and must never leave a torn marker -- ``load_prebuilt_metadata``
reads an unparseable marker as "nothing installed".

No network, no GPU. The release lookup and host detection are monkeypatched; every
install tree is written under the test's own ``tmp_path``. POSIX-only cases (mode bits,
``os.access(X_OK)``, symlinks, ``os.chown``) skip on Windows rather than being weakened,
and the permission cases skip under root, which bypasses the bits they rely on.
"""

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

# build_install writes a real install tree. Imported rather than re-implemented: a private
# copy is how a fixture drifts from the tree the installer actually makes.
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
    """Give every recorded file distinct, non-empty bytes.

    build_install writes the payload libraries empty, and an empty file cannot be
    truncated to half its length or have a byte flipped in it -- the two corruptions
    that separate the size tier from the digest tier would silently become no-ops.
    """
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
    """A healthy install plus a marker written by the REAL write_prebuilt_metadata.

    The fingerprint guard (step 4 of existing_install_current_without_plan) rejects any
    marker it cannot recompute, so a hand-written one would make every "the fast path
    accepts" baseline below pass for the wrong reason.
    """
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

# The digest tier on a Linux install: both copies of each binary, since the root and build/bin
# layouts can rot independently, plus the DiffusionGemma visual server.
HASHED_TIER = (
    "llama-server",
    "llama-quantize",
    "build/bin/llama-server",
    "build/bin/llama-quantize",
    "build/bin/llama-diffusion-gemma-visual-server",
)
# Allowlisted payload matched by lib*.so*: recorded size + mtime_ns, no digest.
SIZE_TIER = ("build/bin/libggml.so", "build/bin/libllama.so")


# ── PART 1: the corruption matrix ────────────────────────────────────────────────────
def test_the_healthy_install_is_accepted_by_the_fast_path(tmp_path, monkeypatch):
    """The baseline every corruption below is measured against.

    Asserted again inside each corruption test before the damage is applied, so a test
    that goes green can only have gone green because of the corruption it names.
    """
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
    """The case the whole two-tier record exists for.

    One bit flipped, byte count identical, mode identical, and the file still parses as
    whatever it was: every structural check in the precheck passes it. Nothing but the
    recorded sha256 separates this install from the one that was downloaded -- and the
    proof is the second half, which drops the digest and watches the same flip sail
    through on size alone.
    """
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
    "relative,still_runs",
    (
        ("llama-server", False),
        ("llama-quantize", False),
        ("build/bin/llama-diffusion-gemma-visual-server", True),
    ),
)
def test_the_execute_bit_is_checked_on_the_runtime_copies_only(
    tmp_path, monkeypatch, relative, still_runs
):
    """Where the fast path stops and the full re-validation takes over on this.

    The precheck's os.access(X_OK) covers install_runtime_dir()/llama-{server,quantize}
    and nothing else, so a root copy or the visual server can lose its execute bit and
    still read as current -- the fast path answers True in every row below.

    _existing_install_runs is the half that differs. It now rejects a root llama-server or
    llama-quantize that cannot be executed, via _damaged_entrypoint; it used to answer True
    for those too, because _binary_image_runs fails OPEN on EACCES (only ENOEXEC and a
    loader-level crash are verdicts). The visual server is not an entrypoint it probes, so
    that row still passes, and the blast radius there is bounded: llama_server_candidates
    keeps scanning inside the same layout and finds the executable build/bin copy.
    """
    install_dir = _install(tmp_path, monkeypatch)
    assert _fast_path(install_dir) is True
    assert LLAMA._existing_install_runs(install_dir, LINUX) is True
    (install_dir / relative).chmod(0o644)
    assert _fast_path(install_dir) is True
    assert LLAMA._existing_install_runs(install_dir, LINUX) is still_runs


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
    """The documented edge of the size tier, asserted so it is a decision and not a surprise.

    runtime_file_records hashes three binaries and stats everything else, because hashing
    300 MB of CUDA kernels on every update is not worth it. So a same-size rewrite of a
    shared library is invisible here. Pre-existing in effect: the full re-validation only
    ever globbed for existence and started llama-server, and a corrupt libggml this file
    can write does not stop a stub from exiting 0 either -- asserted below.
    """
    install_dir = _install(tmp_path, monkeypatch)
    assert _fast_path(install_dir) is True
    target = install_dir / relative
    target.write_bytes(b"\x00" * target.stat().st_size)
    assert _files_match(install_dir) is True
    assert _fast_path(install_dir) is True
    assert LLAMA._existing_install_runs(install_dir, LINUX) is True


# ── PART 2: what the fingerprint does and does not cover ─────────────────────────────
def test_runtime_files_is_not_an_input_to_the_marker_fingerprint(tmp_path, monkeypatch):
    """Establishes the boundary the rest of Part 2 explores.

    _marker_install_fingerprint reads twelve release-identity keys plus the upstream tag.
    runtime_files is not among them, so the self-consistency guard that proves the marker
    "was written whole by this installer" says nothing about the integrity record inside
    it. Not a defect on its own -- the marker is a local unsigned file, and anyone who can
    rewrite runtime_files can delete the marker outright, which costs a re-download rather
    than keeping a broken install -- but it is what makes (a) and (c) below possible.
    """
    install_dir = _install(tmp_path, monkeypatch)
    marker = _marker(install_dir)
    before = LLAMA._marker_install_fingerprint(marker)
    assert before == marker["install_fingerprint"]

    marker["runtime_files"] = {}
    assert LLAMA._marker_install_fingerprint(marker) == before
    marker["runtime_files"] = {"llama-server": {"size": 1}}
    assert LLAMA._marker_install_fingerprint(marker) == before

    # A release-identity key, by contrast, is covered.
    marker["release_tag"] = "release-2"
    assert LLAMA._marker_install_fingerprint(marker) != before


def test_dropping_the_digest_from_one_entry_downgrades_it_to_the_size_tier(tmp_path, monkeypatch):
    """(a) An entry with size but no sha256 is accepted, and checked on size alone.

    _runtime_files_match treats a missing digest as "this file has none to check" --
    the same shape the payload tier legitimately writes -- so the tiers are not carried
    in the marker as a policy, only as the presence of a key. The install is genuinely
    broken afterwards and the fast path keeps it; what makes that a boundary statement
    rather than a bug report is that reaching it needs a write to the marker, and the
    same write could have said anything at all.
    """
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

    # ... and the size half of that entry still bites.
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
    """(c) Removing an entry is not the same as emptying the record.

    The loop iterates what is recorded, and the docstring is explicit that a binary
    "never recorded is not evidence against the install" -- so a dropped entry leaves
    that one file unchecked while every other entry still bites. The second half proves
    the remaining entries are live, which is what makes the first half a scoped statement
    rather than "the record can be switched off one key at a time".
    """
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


# ── PART 3: whisper, which records no payload digests ────────────────────────────────
WHISPER_LINUX = whisper_host(WHISPER.HostInfo)
_GGML_TREE = "ggml-tree-aaaa"


def _whisper_selection(**overrides):
    # CORE, not WHISPER.InstallSelection: this file loads its own prebuilt_core instance and
    # the marker writer accepting it is part of what Part 3 says.
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
    """The premise of the next test, asserted rather than assumed.

    asset_sha256 is the digest of the ARCHIVE, checked once at download time against the
    release's checksum index; it says nothing about what is on disk a month later. The
    marker now also records the extracted payload -- size + sha256 for whisper-server --
    so a later no-network re-check has something to ask.
    """
    install_dir = _whisper_install(tmp_path, monkeypatch)
    marker = _whisper_marker(install_dir)
    assert marker["asset_sha256"] == "c" * 64
    server = WHISPER.installed_server_path(install_dir, WHISPER_LINUX)
    relative = server.relative_to(install_dir).as_posix()
    assert marker["runtime_files"][relative]["sha256"] == CORE.sha256_file(server)
    assert marker["runtime_files"][relative]["size"] == server.stat().st_size


def test_whisper_a_truncated_server_with_bytes_left_is_rejected(tmp_path, monkeypatch):
    """A whisper-server left half-written by a full disk or an interrupted extract.

    It is still a non-empty executable file, which is all the shape checks can see, so
    before the payload record this install was KEPT and the failure surfaced when the
    user pressed the dictation key instead of at update time. The recorded size and
    digest are what turn it into a re-download, which is the repair the user wanted.
    """
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
    # The tree on disk is untouched; only the pairing moved.
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
    """A hardlinked ggml library left as a stub -- most of a CUDA or ROCm pairing's bytes.

    linked_libraries is checked for PRESENCE by name, so before the payload record this
    was "intact" and dictation failed at load time with a dynamic linker error. The
    recorded size catches it for the price of a stat; the paired ggml tree above answers
    the different question of whether llama's runtime moved out from under it.
    """
    install_dir = _whisper_install(tmp_path, monkeypatch, slim = True)
    bin_dir = WHISPER.runtime_bin_dir(install_dir, WHISPER_LINUX)
    (bin_dir / "libggml.so.0").write_bytes(b"\x00")
    assert WHISPER.installed_tree_is_intact(install_dir, WHISPER_LINUX) is False
    assert _whisper_keep(install_dir) is False


# ── PART 4: the marker rewrite as a filesystem operation ─────────────────────────────
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

# release_tag and tag are rendered in the Studio About tab through /api/system/hardware,
# so a rewrite that drops them changes what the UI shows.
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
    """The reason for temp-and-replace at all.

    An in-place rewrite truncates first, so an ENOSPC or an I/O error mid-write strands a
    partial UNSLOTH_*_INFO.json -- and load_prebuilt_metadata reads an unparseable marker
    as "nothing installed", retiring an install that is perfectly fine. Here the swap
    itself fails: the previous marker must be byte-identical afterwards, and no .tmp-*
    may survive inside the install directory, where _swap_into_place would carry it into
    the live tree.
    """
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
    """End to end: the real marker of a real install, through the real rewriter.

    The reuse path rewrites a marker that is already in service, so the rewrite must
    preserve every field the precheck reads back -- the fingerprint inputs, the
    runtime_files record, and the release_tag/tag pair the About tab renders.
    """
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
