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
    monkeypatch.setattr(hf_cache_state, "hf_partials_are_resumable", lambda: True)
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
    monkeypatch.setattr(hf_cache_state, "hf_partials_are_resumable", lambda: False)
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
