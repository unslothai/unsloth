# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The whisper.cpp payload record: can an update keep an install whose bytes rotted?

Before this, a whisper marker recorded nothing about the files it installed. The reuse
predicate (``existing_install_matches``) and the no-network fast path
(``_existing_install_is_intact``) both ask ``installed_tree_is_intact``, and all it could
establish was that ``whisper-server`` was a non-empty file and (off Windows) executable.
So a server truncated by a full disk, or a hardlinked ``libggml-base.so`` left as a stub,
was reported "already matches" and the user met the failure when they pressed the
dictation key. llama.cpp had no such gap: it records ``runtime_files`` (size + sha256 per
runtime file) and compares them in ``_runtime_files_match``.

whisper now records the same key in the same shape. The tests here are the boundary:

  * PART 1 -- what is recorded, and that it is legible beside the llama marker.
  * PART 2 -- the four corruptions that must now be REJECTED, each asserted healthy
    first so a rejection cannot pass for the wrong reason.
  * PART 3 -- backwards compatibility, which is the hard requirement. Every marker
    already on a user's disk lacks the key. It must be KEPT (a re-download is 200-400 MB
    for a tree that is fine), backfilled once under the install lock, and fast after.
  * PART 4 -- the deliberate non-rejections: ``mtime_ns`` moves on a restore from backup
    or a container layer without a byte changing, so it is recorded and never compared.
  * PART 5 -- forward compatibility, proved against a real released module
    (``v0.1.808-beta``), which must ignore a key it has never heard of.

No network, no GPU: the install trees are written under ``tmp_path`` and the only
subprocess reads files extracted from a local git tag. POSIX-only cases skip on Windows
rather than being weakened, and the permission case skips under root.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

WINDOWS_HOST = os.name == "nt"
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0

POSIX_ONLY = pytest.mark.skipif(
    WINDOWS_HOST,
    reason = "mode bits and os.access(X_OK) are POSIX only",
)
NOT_ROOT = pytest.mark.skipif(
    IS_ROOT,
    reason = "root bypasses the permission bits this asserts on",
)

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
STUDIO_DIR = PACKAGE_ROOT / "studio"
if str(STUDIO_DIR) not in sys.path:
    # The installers import each other by module name; spec-based loading needs studio/ reachable.
    sys.path.insert(0, str(STUDIO_DIR))


def _load(module_name: str, filename: str):
    """Load one installer under a name of this file's own, so nothing here can be
    disturbed by -- or disturb -- another test module's instance of it."""
    path = STUDIO_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


WHISPER = _load("studio_install_whisper_prebuilt_pr10648_digests", "install_whisper_prebuilt.py")
LLAMA = _load("studio_install_llama_prebuilt_pr10648_digests", "install_llama_prebuilt.py")

RELEASE_TAG = "v1.9.1-unsloth.1"
GGML_TREE = "ggml-tree-aaaa"
SERVER_BYTES = b"#!/bin/sh\necho whisper-server\nexit 0\n"


def _host(**overrides):
    fields = dict(
        system = "Linux",
        machine = "x64",
        whisper_os = "linux",
        whisper_arch = "x64",
        archive_ext = ".tar.gz",
        is_windows = False,
        is_macos = False,
        is_apple_silicon = False,
    )
    fields.update(overrides)
    return WHISPER.HostInfo(**fields)


LINUX = _host()


def _selection(**overrides):
    fields = dict(
        published_repo = WHISPER.DEFAULT_PUBLISHED_REPO,
        release_tag = RELEASE_TAG,
        upstream_tag = "v1.9.1",
        source_commit = "0" * 40,
        asset = f"whisper-{RELEASE_TAG}-linux-x64-cpu.tar.gz",
        asset_sha256 = "c" * 64,
        backend = "cpu",
        runtime_line = None,
        coverage = {"min_os": None},
        studio_protocol = "inference/multipart-v1",
        platform_os = "linux",
        platform_arch = "x64",
    )
    fields.update(overrides)
    return WHISPER.InstallSelection(**fields)


SLIM_LIBRARIES = ("libggml.so.0", "libggml-base.so.0")


def _install(
    tmp_path: Path,
    monkeypatch,
    *,
    slim: bool = False,
) -> Path:
    """A whisper install tree plus a marker written by the REAL writer, so the record
    under test is the one an install actually produces -- not a fixture's idea of it."""
    install_dir = tmp_path / "whisper.cpp"
    bin_dir = WHISPER.runtime_bin_dir(install_dir, LINUX)
    bin_dir.mkdir(parents = True)
    server = WHISPER.installed_server_path(install_dir, LINUX)
    server.write_bytes(SERVER_BYTES)
    server.chmod(0o755)
    (bin_dir / "libwhisper.so.1").write_bytes(b"dummy-libwhisper")
    if slim:
        for name in SLIM_LIBRARIES:
            (bin_dir / name).write_bytes(b"ggml-payload-" + name.encode("utf-8"))
    # The live llama marker a slim bundle is wired against; never the real one on this box.
    monkeypatch.setattr(WHISPER, "installed_llama_ggml_tree", lambda *_a, **_k: GGML_TREE)
    WHISPER.write_prebuilt_metadata(install_dir, _selection(**_slim_fields(tmp_path, slim)))
    return install_dir


def _slim_fields(tmp_path: Path, slim: bool) -> dict:
    if not slim:
        return {}
    return dict(
        install_kind = "slim",
        paired_llama_tag = "b9001",
        linked_from = str(tmp_path / "llama.cpp" / "build" / "bin"),
        linked_libraries = SLIM_LIBRARIES,
        runtime_wiring_version = WHISPER.SLIM_RUNTIME_WIRING_VERSION,
        linked_runtime_directories = (),
    )


def _marker_path(install_dir: Path) -> Path:
    return install_dir / WHISPER.METADATA_FILENAME


def _marker(install_dir: Path) -> dict:
    return json.loads(_marker_path(install_dir).read_text(encoding = "utf-8"))


def _rewrite_marker(install_dir: Path, marker: dict) -> None:
    _marker_path(install_dir).write_text(json.dumps(marker, indent = 2), encoding = "utf-8")


def _intact(install_dir: Path) -> bool:
    """The shared on-disk half, which both decision paths below delegate to."""
    return WHISPER.installed_tree_is_intact(install_dir, LINUX)


def _reuse(
    install_dir: Path,
    *,
    slim: bool = False,
    tmp_path: Path | None = None,
) -> bool:
    """The ordinary reuse predicate: "this release is already installed, skip the download"."""
    return WHISPER.existing_install_matches(
        install_dir, LINUX, _selection(**_slim_fields(tmp_path or install_dir.parent, slim))
    )


def _fast_path(install_dir: Path) -> bool:
    """The no-network fast path: "do not even ask which release is newest"."""
    return (
        WHISPER._existing_install_is_intact(
            install_dir,
            LINUX,
            published_repo = WHISPER.DEFAULT_PUBLISHED_REPO,
            requested_backend = "cpu",
        )
        is not None
    )


def _server_key(install_dir: Path) -> str:
    return WHISPER.installed_server_path(install_dir, LINUX).relative_to(install_dir).as_posix()


# ── PART 1: what the marker records ──────────────────────────────────────────────────
def test_the_marker_records_the_server_it_installed(tmp_path, monkeypatch):
    """The update path decides "already matches" without ever starting whisper-server,
    so the recorded digest is what stands in for "it launched once"."""
    install_dir = _install(tmp_path, monkeypatch)
    server = WHISPER.installed_server_path(install_dir, LINUX)
    record = _marker(install_dir)["runtime_files"][_server_key(install_dir)]
    assert record["size"] == len(SERVER_BYTES)
    assert record["sha256"] == WHISPER.sha256_file(server)
    assert isinstance(record["mtime_ns"], int)
    # Relative POSIX keys, so a marker stays valid when the install dir is moved or read
    # on a host that spells separators the other way.
    assert all(not Path(key).is_absolute() for key in _marker(install_dir)["runtime_files"])


def test_a_slim_marker_records_the_ggml_libraries_it_hardlinked(tmp_path, monkeypatch):
    """A slim bundle ships no ggml of its own: those hardlinks ARE most of its bytes on a
    CUDA or ROCm pairing, and nothing else in the marker describes them."""
    install_dir = _install(tmp_path, monkeypatch, slim = True)
    records = _marker(install_dir)["runtime_files"]
    for name in SLIM_LIBRARIES:
        library = WHISPER.runtime_bin_dir(install_dir, LINUX) / name
        record = records[library.relative_to(install_dir).as_posix()]
        assert record["size"] == library.stat().st_size
        # Size tier only: hashing hundreds of MB of kernels on every update is not worth it.
        assert "sha256" not in record


def test_the_record_has_the_same_shape_as_the_llama_one(tmp_path, monkeypatch):
    """The two markers sit side by side in one Unsloth home and are read by the same
    people; a second spelling of the same fact is how one of them comes to drift."""
    install_dir = _install(tmp_path, monkeypatch)
    server = WHISPER.installed_server_path(install_dir, LINUX)
    assert WHISPER._stat_record(server).keys() == LLAMA._stat_record(server).keys()
    record = _marker(install_dir)["runtime_files"][_server_key(install_dir)]
    assert set(record) == {"size", "mtime_ns", "sha256"}


# ── PART 2: the corruptions that must now be rejected ────────────────────────────────
def test_a_truncated_server_is_rejected(tmp_path, monkeypatch):
    """A whisper-server left half-written by a full disk or an interrupted extract.

    It is still a non-empty executable file, so every shape check passes it; before the
    record this install was kept and dictation failed at the user's next keypress.
    """
    install_dir = _install(tmp_path, monkeypatch)
    assert _intact(install_dir) is True and _reuse(install_dir) is True
    server = WHISPER.installed_server_path(install_dir, LINUX)
    server.write_bytes(SERVER_BYTES[: len(SERVER_BYTES) // 2])
    assert server.stat().st_size > 0
    assert WINDOWS_HOST or os.access(server, os.X_OK)
    assert _intact(install_dir) is False
    assert _reuse(install_dir) is False
    assert _fast_path(install_dir) is False


def test_a_same_size_byte_flip_is_caught_only_by_the_digest(tmp_path, monkeypatch):
    """Bit rot, a bad cable or a partial overwrite: the file is exactly as long as it was.

    This is the one corruption the size tier cannot see, and the reason whisper-server is
    hashed rather than statted like the ggml libraries beside it.
    """
    install_dir = _install(tmp_path, monkeypatch)
    assert _intact(install_dir) is True
    server = WHISPER.installed_server_path(install_dir, LINUX)
    before = server.stat().st_size
    data = bytearray(server.read_bytes())
    data[len(data) // 2] ^= 0xFF
    server.write_bytes(bytes(data))
    assert server.stat().st_size == before, "the flip must preserve the size or it proves nothing"
    recorded = _marker(install_dir)["runtime_files"][_server_key(install_dir)]
    assert recorded["size"] == server.stat().st_size
    assert _intact(install_dir) is False
    assert _reuse(install_dir) is False
    assert _fast_path(install_dir) is False


def test_a_deleted_server_is_rejected(tmp_path, monkeypatch):
    """A recorded file that vanished is evidence against the install; the shape check
    caught this one already, and the record must not soften it."""
    install_dir = _install(tmp_path, monkeypatch)
    assert _intact(install_dir) is True
    WHISPER.installed_server_path(install_dir, LINUX).unlink()
    assert _intact(install_dir) is False
    assert _reuse(install_dir) is False
    assert _fast_path(install_dir) is False


def test_a_corrupt_wired_library_is_rejected(tmp_path, monkeypatch):
    """A hardlinked ggml library left as a stub. linked_libraries is checked for PRESENCE
    by name, so a one-byte libggml.so.0 used to read as intact wiring and the sidecar
    failed with a dynamic linker error the user could do nothing with."""
    install_dir = _install(tmp_path, monkeypatch, slim = True)
    assert _intact(install_dir) is True
    assert _reuse(install_dir, slim = True, tmp_path = tmp_path) is True
    library = WHISPER.runtime_bin_dir(install_dir, LINUX) / "libggml.so.0"
    library.write_bytes(b"\x00")
    assert library.is_file()
    assert _intact(install_dir) is False
    assert _reuse(install_dir, slim = True, tmp_path = tmp_path) is False
    assert _fast_path(install_dir) is False


def test_a_same_size_flip_in_a_ggml_library_is_deliberately_not_caught(tmp_path, monkeypatch):
    """The honest limit of the size tier, asserted so nobody reads the test above as more
    than it is: the libraries are statted, not hashed, because on a CUDA pairing they are
    hundreds of MB and this check runs on every update. Truncation and deletion -- what a
    full disk, an interrupted extract or a half-removed llama install actually produce --
    are what it is built to catch."""
    install_dir = _install(tmp_path, monkeypatch, slim = True)
    library = WHISPER.runtime_bin_dir(install_dir, LINUX) / "libggml.so.0"
    data = bytearray(library.read_bytes())
    data[0] ^= 0xFF
    library.write_bytes(bytes(data))
    assert _intact(install_dir) is True


def test_a_marker_whose_record_is_unusable_fails_closed(tmp_path, monkeypatch):
    """Present but unreadable is not the same as absent. The record is the evidence that
    replaces starting the binary, so an empty or malformed one proves nothing -- and it
    cannot have come from the writer, which omits the key rather than writing {}."""
    install_dir = _install(tmp_path, monkeypatch)
    for damaged in ({}, [], "runtime", {_server_key(install_dir): "not-a-record"}):
        marker = _marker(install_dir)
        marker["runtime_files"] = damaged
        _rewrite_marker(install_dir, marker)
        assert _intact(install_dir) is False, damaged
        assert _fast_path(install_dir) is False, damaged


@POSIX_ONLY
@NOT_ROOT
def test_an_unreadable_server_is_rejected_rather_than_skipped(tmp_path, monkeypatch):
    """A tree half-owned by a previous sudo install: the digest cannot be computed, and
    "could not check" must not read as "checked and fine"."""
    install_dir = _install(tmp_path, monkeypatch)
    assert _intact(install_dir) is True
    server = WHISPER.installed_server_path(install_dir, LINUX)
    server.chmod(0o000)
    try:
        assert _intact(install_dir) is False
    finally:
        server.chmod(0o755)
    assert _intact(install_dir) is True


# ── PART 3: backwards compatibility, the hard requirement ────────────────────────────
def _legacy(install_dir: Path) -> dict:
    """The marker every existing user has: byte-identical to what this installer writes,
    minus the key that did not exist when it was written."""
    marker = _marker(install_dir)
    marker.pop("runtime_files", None)
    _rewrite_marker(install_dir, marker)
    assert "runtime_files" not in _marker(install_dir)
    return marker


def test_a_marker_written_before_the_record_is_kept_not_re_downloaded(tmp_path, monkeypatch):
    """A user upgrading Unsloth Studio over an install made by an older one.

    Nothing is wrong with their tree; it simply predates the key. The reuse predicate is
    what decides whether the update downloads 200-400 MB again, so it must say yes. The
    no-network fast path is allowed to decline -- that costs one release lookup, once --
    and that is what settles the record.
    """
    install_dir = _install(tmp_path, monkeypatch)
    _legacy(install_dir)
    assert _intact(install_dir) is True
    assert _reuse(install_dir) is True
    assert _fast_path(install_dir) is False
    assert WHISPER.kept_install_needs_settling(install_dir) is True


def test_the_legacy_marker_is_backfilled_once_and_fast_after(tmp_path, monkeypatch):
    """The other half: "takes the full path once" is only true if something writes the
    record. Otherwise every single update re-fetches the release, its manifest and its
    checksum index forever."""
    install_dir = _install(tmp_path, monkeypatch)
    _legacy(install_dir)
    WHISPER.settle_kept_install(install_dir)

    server = WHISPER.installed_server_path(install_dir, LINUX)
    record = _marker(install_dir)["runtime_files"][_server_key(install_dir)]
    assert record["sha256"] == WHISPER.sha256_file(server)
    assert record["size"] == server.stat().st_size
    assert _fast_path(install_dir) is True
    assert WHISPER.kept_install_needs_settling(install_dir) is False
    # And the record it just wrote is load-bearing from here on.
    server.write_bytes(SERVER_BYTES[:4])
    assert _intact(install_dir) is False


def test_the_backfill_records_the_live_bytes_of_a_slim_tree(tmp_path, monkeypatch):
    """A legacy SLIM install: the wiring it hardlinked has to be recorded too, or the
    backfill would leave the half of the payload that is most of the bytes unprotected."""
    install_dir = _install(tmp_path, monkeypatch, slim = True)
    _legacy(install_dir)
    WHISPER.settle_kept_install(install_dir)
    records = _marker(install_dir)["runtime_files"]
    for name in SLIM_LIBRARIES:
        library = WHISPER.runtime_bin_dir(install_dir, LINUX) / name
        assert records[library.relative_to(install_dir).as_posix()]["size"] == (
            library.stat().st_size
        )
    (WHISPER.runtime_bin_dir(install_dir, LINUX) / SLIM_LIBRARIES[0]).write_bytes(b"\x00")
    assert _intact(install_dir) is False


def test_the_backfill_never_rewrites_a_record_it_did_not_take(tmp_path, monkeypatch):
    """Added, never corrected. Overwriting a present record with whatever is on disk now
    would re-bless bytes no run ever hashed -- it would turn the guard into a rubber
    stamp the first time an update ran over a corrupt tree."""
    install_dir = _install(tmp_path, monkeypatch)
    recorded = _marker(install_dir)["runtime_files"]
    WHISPER.installed_server_path(install_dir, LINUX).write_bytes(b"replaced-behind-our-back")
    WHISPER.settle_kept_install(install_dir)
    assert _marker(install_dir)["runtime_files"] == recorded
    assert _intact(install_dir) is False


@POSIX_ONLY
def test_the_backfill_replaces_the_marker_atomically(tmp_path, monkeypatch):
    """The marker is already in service: a torn write reads as "nothing installed" and
    costs the user the whole install. Temp-and-replace, so a failure part-way leaves the
    marker that was there, the mode survives for a group-shared install, and no .tmp-*
    sibling is left behind."""
    install_dir = _install(tmp_path, monkeypatch)
    _legacy(install_dir)
    marker_path = _marker_path(install_dir)
    marker_path.chmod(0o644)
    before = marker_path.stat()

    WHISPER.settle_kept_install(install_dir)

    after = marker_path.stat()
    # A different inode is the observable signature of temp-and-replace: the live marker
    # was swapped, never truncated in place.
    assert after.st_ino != before.st_ino
    assert after.st_mode == before.st_mode
    assert [p.name for p in install_dir.iterdir() if ".tmp-" in p.name] == []
    assert _marker(install_dir)["runtime_files"]


def test_an_upgrade_does_not_re_download_and_settles_under_the_lock(tmp_path, monkeypatch):
    """End to end over the real keep path: an older Studio's install, one `studio update`.

    install_selected_prebuilt is where a keep becomes a download, so this is the test
    that actually proves the upgrade is free. The backfill is a read-modify-write of a
    live marker, so it must happen under the install lock -- outside it, a concurrent
    installer swapping in a new release has its fresh marker overwritten with the old
    release's fields.
    """
    install_dir = _install(tmp_path, monkeypatch)
    _legacy(install_dir)
    selection = _selection()
    bundle = WHISPER.ReleaseBundle(
        repo = WHISPER.DEFAULT_PUBLISHED_REPO,
        release_tag = RELEASE_TAG,
        manifest = {},
        asset_urls = {},
    )

    downloads = {"n": 0}

    def no_downloads(*_args, **_kwargs):
        downloads["n"] += 1
        raise AssertionError("the upgrade re-installed instead of keeping the existing tree")

    depth = {"now": 0, "backfills_under_lock": 0, "backfills_outside": 0}
    real_lock = WHISPER.install_lock
    real_backfill = WHISPER._backfill_runtime_file_records

    import contextlib

    @contextlib.contextmanager
    def counting_lock(path):
        with real_lock(path):
            depth["now"] += 1
            try:
                yield
            finally:
                depth["now"] -= 1

    def counting_backfill(directory):
        key = "backfills_under_lock" if depth["now"] else "backfills_outside"
        depth[key] += 1
        real_backfill(directory)

    monkeypatch.setattr(WHISPER, "_install_from_bundle", no_downloads)
    monkeypatch.setattr(WHISPER, "install_lock", counting_lock)
    monkeypatch.setattr(WHISPER, "_backfill_runtime_file_records", counting_backfill)

    result = WHISPER.core.install_selected_prebuilt(
        WHISPER._OPS,
        install_dir,
        host = LINUX,
        bundle = bundle,
        selection = selection,
        force = False,
    )
    assert result == WHISPER.EXIT_SUCCESS
    assert downloads["n"] == 0
    # The injection fired: the settle really went through the patched lock and backfill.
    assert depth == {"now": 0, "backfills_under_lock": 1, "backfills_outside": 0}
    assert _marker(install_dir)["runtime_files"]

    # Nothing left to settle: the next update takes no lock to write.
    depth["backfills_under_lock"] = 0
    assert (
        WHISPER.core.install_selected_prebuilt(
            WHISPER._OPS,
            install_dir,
            host = LINUX,
            bundle = bundle,
            selection = selection,
            force = False,
        )
        == WHISPER.EXIT_SUCCESS
    )
    assert depth["backfills_under_lock"] == 0 and depth["backfills_outside"] == 0
    assert downloads["n"] == 0


def test_a_backfill_that_cannot_hash_writes_nothing(tmp_path, monkeypatch):
    """A scanner holding whisper-server open, or a tree half-owned by root. An empty
    record is not evidence, and writing one would fail closed on every later update --
    which is the re-download loop this whole part exists to prevent."""
    install_dir = _install(tmp_path, monkeypatch)
    _legacy(install_dir)
    fired = {"n": 0}

    def unreadable(_path):
        fired["n"] += 1
        raise OSError("Input/output error")

    monkeypatch.setattr(WHISPER, "sha256_file", unreadable)
    WHISPER.settle_kept_install(install_dir)
    assert fired["n"] > 0, "the injected hasher never ran; the test proves nothing"
    assert "runtime_files" not in _marker(install_dir)
    # Still kept, still no download: it simply stays on the full path.
    assert _intact(install_dir) is True
    assert _reuse(install_dir) is True


# ── PART 4: what must deliberately NOT reject ────────────────────────────────────────
def test_a_changed_mtime_alone_does_not_reject(tmp_path, monkeypatch):
    """A restore from backup, an rsync, a container layer or a `tar -x` without
    --touch: the timestamps move, not a byte changes. mtime_ns is recorded (it is what
    makes the size tier cheap to reason about) and never compared, because the answer to
    a mismatch here is a 200-400 MB re-download."""
    install_dir = _install(tmp_path, monkeypatch, slim = True)
    recorded = _marker(install_dir)["runtime_files"]
    for relative in recorded:
        target = install_dir / relative
        os.utime(target, ns = (recorded[relative]["mtime_ns"] + 10**9,) * 2)
        assert target.stat().st_mtime_ns != recorded[relative]["mtime_ns"]
    assert _intact(install_dir) is True
    assert _reuse(install_dir, slim = True, tmp_path = tmp_path) is True
    assert _fast_path(install_dir) is True


def test_an_unrecorded_file_appearing_beside_the_payload_does_not_reject(tmp_path, monkeypatch):
    """A user's own notes, an antivirus quarantine stub, a .bak from a support session.
    A file that appeared since is not evidence against the install; only a RECORDED file
    that moved is."""
    install_dir = _install(tmp_path, monkeypatch)
    (WHISPER.runtime_bin_dir(install_dir, LINUX) / "notes.txt").write_text("hello")
    assert _intact(install_dir) is True
    assert _fast_path(install_dir) is True


# ── PART 5: forward compatibility against a really released module ───────────────────
_OLD_TAG = "v0.1.808-beta"
_OLD_MODULES = ("install_whisper_prebuilt.py", "prebuilt_core.py", "install_llama_prebuilt.py")

_OLD_READER = """
import dataclasses, json, pathlib, sys
import install_whisper_prebuilt as M

spec = json.loads(sys.argv[1])
install_dir = pathlib.Path(spec["install_dir"])
host = M.HostInfo(**{
    key: value
    for key, value in spec["host"].items()
    if key in {f.name for f in dataclasses.fields(M.HostInfo)}
})
selection = M.InstallSelection(**{
    key: value
    for key, value in spec["selection"].items()
    if key in {f.name for f in dataclasses.fields(M.InstallSelection)}
})
source = pathlib.Path(M.__file__).read_text(encoding = "utf-8")
print(json.dumps({
    "module_file": M.__file__,
    "core_file": M.core.__file__,
    "knows_the_key": "runtime_files" in source,
    "matches": bool(M.existing_install_matches(install_dir, host, selection)),
    "marker_keys": sorted(M.load_prebuilt_metadata(install_dir) or {}),
}))
"""


def _extract_released_studio(tmp_path: Path) -> Path:
    """The three installer modules exactly as `{tag}` shipped them, from the local git
    tag. Read-only, no network, and skipped rather than faked where git cannot answer."""
    old_dir = tmp_path / "released_studio"
    old_dir.mkdir()
    for name in _OLD_MODULES:
        try:
            blob = subprocess.run(
                ["git", "show", f"{_OLD_TAG}:studio/{name}"],
                cwd = PACKAGE_ROOT,
                capture_output = True,
                timeout = 60,
            )
        except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - CI without git
            pytest.skip(f"git is unavailable here: {exc}")
        if blob.returncode != 0 or not blob.stdout:  # pragma: no cover - shallow checkout
            pytest.skip(f"{_OLD_TAG}:studio/{name} is not in this checkout")
        (old_dir / name).write_bytes(blob.stdout)
    return old_dir


def test_a_released_older_studio_ignores_the_new_key(tmp_path, monkeypatch):
    """Two Unsloth Studios can share one UNSLOTH_HOME -- an older one pinned in a venv, a
    rollback, a second checkout. The older one must read a marker carrying a key it has
    never heard of and keep the install, not refuse it and re-download 200-400 MB.

    Only ADDING a key makes this true, so it is asserted against the module v0.1.808-beta
    actually shipped rather than argued from the diff. A subprocess, so the old module's
    sys.path insert and its instance of prebuilt_core cannot leak into this process.
    """
    install_dir = _install(tmp_path, monkeypatch)
    assert "runtime_files" in _marker(install_dir), "nothing forward-compatible to test"
    old_dir = _extract_released_studio(tmp_path)

    payload = json.dumps(
        {
            "install_dir": str(install_dir),
            "host": dataclasses.asdict(LINUX),
            "selection": {
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in dataclasses.asdict(_selection()).items()
            },
        }
    )
    env = dict(os.environ)
    # The old copies first, the live studio/ behind them for the support packages those
    # modules import (backend.utils.prebuilt.*), which are not part of this marker's story.
    env["PYTHONPATH"] = os.pathsep.join([str(old_dir), str(STUDIO_DIR)])
    completed = subprocess.run(
        [sys.executable, "-c", _OLD_READER, payload],
        cwd = str(old_dir),
        capture_output = True,
        text = True,
        timeout = 300,
        env = env,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    # The injection fired: this really is the released module, not the one under test.
    assert result["module_file"].startswith(str(old_dir))
    assert result["core_file"].startswith(str(old_dir))
    assert result["knows_the_key"] is False
    # It read the whole marker, new key included, and kept the install.
    assert "runtime_files" in result["marker_keys"]
    assert result["matches"] is True
