# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A row may only offer Resume for a partial that can actually be reopened.

The installed huggingface_hub cannot answer this alone. A cache shared with a newer
environment holds ``<etag>.<nonce>.incomplete`` files that even a resuming writer will not
reopen, and this repo's own pins produce exactly that mix: Python 3.10+ takes hub >= 1.23,
older takes 0.36.2, one cache between them. Deriving the answer from transport plus installed
version promised a resume there and then purged the bytes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from hub.utils import download_manifest, download_registry, hf_cache_state, state_dir
from hub.utils.download_manifest import ExpectedFile
from hub.utils.inventory_scan import partial_resume_available


BLOB = "a" * 64
LEGACY_PARTIAL = f"{BLOB}{hf_cache_state.INCOMPLETE_SUFFIX}"
NONCE_PARTIAL = f"{BLOB}.deadbeef{hf_cache_state.INCOMPLETE_SUFFIX}"


@pytest.fixture
def cache(monkeypatch, tmp_path):
    """A repo cache with a legacy huggingface_hub installed, so only the per-file check can
    reject a partial."""
    root = tmp_path / "hub"
    entry = root / "models--Org--Model"
    (entry / "blobs").mkdir(parents = True)
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    for module in (download_registry, hf_cache_state):
        monkeypatch.setattr(module, "hf_cache_root", lambda **_kw: root, raising = False)
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda *_a, **_k: [root])
    monkeypatch.setattr(hf_cache_state, "hf_partials_are_resumable", lambda _root = None: True)
    return entry


def _record(transport: str, entry: Path):
    """What a worker leaves behind: the manifest the row reads, and the marker naming the
    writer that produced the partial."""
    download_manifest.write_manifest(
        "model",
        "Org/Model",
        "Q4_K_M",
        [ExpectedFile(path = "model-Q4_K_M.gguf", size = 4096, sha256 = BLOB)],
        transport,
    )
    download_registry._write_marker(entry, transport, "Q4_K_M")


def _partial(entry: Path, name: str):
    (entry / "blobs" / name).write_bytes(b"x" * 128)


def test_a_reopenable_http_partial_can_be_resumed(cache):
    _record("http", cache)
    _partial(cache, LEGACY_PARTIAL)
    assert partial_resume_available("model", "Org/Model", "Q4_K_M") is True


def test_a_nonce_partial_cannot_be_resumed_by_any_writer(cache):
    """The bug: hub <= 1.17 is installed and the marker says HTTP, so a version-only check
    said Resume. This file was written by a 1.18+ client and will be purged, not continued."""
    _record("http", cache)
    _partial(cache, NONCE_PARTIAL)
    assert partial_resume_available("model", "Org/Model", "Q4_K_M") is False


def test_a_xet_partial_is_never_resumable(cache):
    _record("xet", cache)
    _partial(cache, LEGACY_PARTIAL)
    assert partial_resume_available("model", "Org/Model", "Q4_K_M") is False


def test_a_row_with_no_partial_left_has_nothing_to_resume(cache):
    _record("http", cache)
    assert partial_resume_available("model", "Org/Model", "Q4_K_M") is False


def test_a_writer_that_cannot_reopen_anything_never_promises_a_resume(cache, monkeypatch):
    """The ordinary install: hub >= 1.18 refetches from zero, so even a legacy-named
    survivor is litter."""
    monkeypatch.setattr(hf_cache_state, "hf_partials_are_resumable", lambda _root = None: False)
    _record("http", cache)
    _partial(cache, LEGACY_PARTIAL)
    assert partial_resume_available("model", "Org/Model", "Q4_K_M") is False


def test_an_unmarked_partial_is_not_resumable(cache):
    """No manifest and no marker: nothing attributes the partial to a writer that resumes."""
    _partial(cache, LEGACY_PARTIAL)
    assert partial_resume_available("model", "Org/Model", "Q4_K_M") is False


def test_a_siblings_resumable_partial_does_not_speak_for_this_one(cache):
    """Two quants of one repo: this one's partial is nonce-named and doomed, the sibling's is
    reopenable. The blob scan behind is_resumable_partial is repo-wide, so the sibling's bytes
    must not answer for this row."""
    _record("http", cache)
    _partial(cache, NONCE_PARTIAL)
    sibling = "b" * 64
    download_manifest.write_manifest(
        "model",
        "Org/Model",
        "Q8_0",
        [ExpectedFile(path = "model-Q8_0.gguf", size = 4096, sha256 = sibling)],
        "http",
    )
    download_registry._write_marker(cache, "http", "Q8_0")
    _partial(cache, f"{sibling}{hf_cache_state.INCOMPLETE_SUFFIX}")

    assert partial_resume_available("model", "Org/Model", "Q8_0") is True
    assert partial_resume_available("model", "Org/Model", "Q4_K_M") is False


COMPANION = "c" * 64


def _record_with_companion(transport: str, entry: Path):
    """A vision quant: its own shard plus the shared mmproj, which every sibling downloads."""
    download_manifest.write_manifest(
        "model",
        "Org/Model",
        "Q4_K_M",
        [
            ExpectedFile(path = "model-Q4_K_M.gguf", size = 4096, sha256 = BLOB),
            ExpectedFile(path = "mmproj-F16.gguf", size = 2048, sha256 = COMPANION),
        ],
        transport,
    )
    download_registry._write_marker(entry, transport, "Q4_K_M")


def test_a_companion_partial_from_another_transport_is_not_a_resume(cache):
    """The companion is governed by .transport.companion, not by this variant's marker.
    prepare_cache_for_transport purges it on an HTTP run, so it cannot back a Resume."""
    _record_with_companion("http", cache)
    download_registry._write_companion_marker(cache, "xet")
    _partial(cache, f"{COMPANION}{hf_cache_state.INCOMPLETE_SUFFIX}")
    assert partial_resume_available("model", "Org/Model", "Q4_K_M") is False


def test_a_companion_partial_written_by_http_still_resumes(cache):
    _record_with_companion("http", cache)
    download_registry._write_companion_marker(cache, "http")
    _partial(cache, f"{COMPANION}{hf_cache_state.INCOMPLETE_SUFFIX}")
    assert partial_resume_available("model", "Org/Model", "Q4_K_M") is True


def test_an_unmarked_companion_partial_is_not_a_resume(cache):
    _record_with_companion("http", cache)
    _partial(cache, f"{COMPANION}{hf_cache_state.INCOMPLETE_SUFFIX}")
    assert partial_resume_available("model", "Org/Model", "Q4_K_M") is False


def test_the_variants_own_partial_still_decides_for_the_variant(cache):
    """A companion marker disagreeing says nothing about the main shard, which this
    variant's own marker vouches for."""
    _record_with_companion("http", cache)
    download_registry._write_companion_marker(cache, "xet")
    _partial(cache, LEGACY_PARTIAL)
    assert partial_resume_available("model", "Org/Model", "Q4_K_M") is True


# --------------------------------------------------------------------------------------------
# One repo can own several active cache directories at once: a case-sensitive filesystem holds
# models--Org--Model beside models--org--model, and every one of them matches. The blob scan
# unions them while a marker read answers from whichever comes first, so a verdict built from
# the two separately can pair a partial in one directory with a marker from another.
# prepare_cache_for_transport judges each directory on its own, so the pairing has to as well.
# --------------------------------------------------------------------------------------------


@pytest.fixture
def split_cache(monkeypatch, tmp_path):
    """Two active entries for one repo. The names are stand-ins so the test also runs on a
    case-insensitive filesystem, where the real pair cannot coexist."""
    root = tmp_path / "hub"
    first = root / "models--Org--Model"
    second = root / "models--Org--Model.case-variant"
    for entry in (first, second):
        (entry / "blobs").mkdir(parents = True)
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(
        download_registry, "iter_active_repo_cache_dirs", lambda *_a, **_k: iter((first, second))
    )
    monkeypatch.setattr(hf_cache_state, "hf_partials_are_resumable", lambda _root = None: True)
    return first, second


def test_a_marker_does_not_vouch_for_another_directorys_companion(split_cache):
    first, second = split_cache
    _record_with_companion("http", first)
    download_registry._write_companion_marker(first, "http")
    # The partial lives next door, under a Xet companion marker that will purge it.
    download_registry._write_marker(second, "http", "Q4_K_M")
    download_registry._write_companion_marker(second, "xet")
    _partial(second, f"{COMPANION}{hf_cache_state.INCOMPLETE_SUFFIX}")

    assert partial_resume_available("model", "Org/Model", "Q4_K_M") is False


def test_a_marker_does_not_vouch_for_another_directorys_main_partial(split_cache):
    first, second = split_cache
    _record("http", first)
    # Same repo, other directory, written by Xet: its own marker is what decides.
    download_registry._write_marker(second, "xet", "Q4_K_M")
    _partial(second, LEGACY_PARTIAL)

    assert partial_resume_available("model", "Org/Model", "Q4_K_M") is False


def test_a_partial_and_its_own_directorys_marker_still_resume(split_cache):
    first, second = split_cache
    _record("http", first)
    download_registry._write_marker(second, "http", "Q4_K_M")
    _partial(second, LEGACY_PARTIAL)

    assert partial_resume_available("model", "Org/Model", "Q4_K_M") is True


# --------------------------------------------------------------------------------------------
# A row is not always displayed from the active cache. local_inventory enumerates remembered
# ("previous HF cache") and custom roots too, and hands each row's own directory down. That
# root holds its own partials and its own manifest scope (state_dir keys manifests by a
# per-cache digest), so the resume verdict has to be asked of it and not of the active root.
# --------------------------------------------------------------------------------------------


@pytest.fixture
def two_roots(monkeypatch, tmp_path):
    """The active hub cache plus a remembered one, both holding the same repo."""
    active = tmp_path / "hub"
    previous = tmp_path / "old" / "hub"
    for root in (active, previous):
        (root / "models--Org--Model" / "blobs").mkdir(parents = True)
    monkeypatch.setenv("HF_HUB_CACHE", str(active))
    monkeypatch.setattr(state_dir, "cache_root", lambda: tmp_path / "state")
    monkeypatch.setattr(hf_cache_state, "hf_partials_are_resumable", lambda _root = None: True)
    for module in (download_registry, hf_cache_state):
        monkeypatch.setattr(
            module,
            "hf_cache_root",
            lambda root = None, **_kw: Path(root) if root is not None else active,
            raising = False,
        )
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda *_a, **_k: [active, previous])
    return active / "models--Org--Model", previous / "models--Org--Model"


def _record_in(
    entry: Path,
    transport: str,
    variant,
    blob: str = BLOB,
):
    download_manifest.write_manifest(
        "model",
        "Org/Model",
        variant,
        [ExpectedFile(path = f"model-{variant or 'main'}.gguf", size = 4096, sha256 = blob)],
        transport,
        hub_cache = entry.parent,
    )
    download_registry._write_marker(entry, transport, variant)


def test_another_roots_partial_does_not_vouch_for_this_row(two_roots):
    """The displayed row lives in the remembered cache and its partial is nonce-named, so
    nothing can reopen it. The active root's legacy partial for the same repo must not be
    what turns this row into "Resume with HTTP to keep the progress you already have"."""
    active, previous = two_roots
    _record_in(active, "http", None)
    _partial(active, LEGACY_PARTIAL)
    _record_in(previous, "http", None)
    _partial(previous, NONCE_PARTIAL)

    assert partial_resume_available("model", "Org/Model", None, previous) is False


def test_a_remembered_roots_own_partial_still_resumes(two_roots):
    """And the other direction: the row's bytes are reopenable and its manifest lives in that
    root's scope, so the verdict may not be lost just because the active root is empty."""
    active, previous = two_roots
    _record_in(previous, "http", "Q4_K_M")
    _partial(previous, LEGACY_PARTIAL)

    assert partial_resume_available("model", "Org/Model", "Q4_K_M", previous) is True
