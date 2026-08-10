# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A partial no huggingface_hub can append to is litter, not resume state."""

import pytest

from hub.utils import download_registry, hf_cache_state


_MAIN = "a" * 64
_PEER = "b" * 64


@pytest.fixture
def blobs(monkeypatch, tmp_path):
    root = tmp_path / "hub"
    entry = root / "models--Org--Model"
    blobs_dir = entry / "blobs"
    blobs_dir.mkdir(parents = True)
    monkeypatch.setenv("HF_HUB_CACHE", str(root))
    monkeypatch.setattr(download_registry, "hf_cache_root", lambda **_kwargs: root)
    return blobs_dir


def _prepare(**kwargs):
    return download_registry.prepare_cache_for_transport(
        "model",
        "Org/Model",
        download_registry.TRANSPORT_HTTP,
        "Q4_K_M",
        only_blob_hashes = frozenset({_MAIN}),
        **kwargs,
    )


@pytest.mark.parametrize(
    "hf_version, resumable",
    [
        ("0.36.2", True),
        ("1.17.0", True),
        ("1.18.0", False),
        ("1.23.0", False),
        ("1.27.0", False),
        ("2.0.0.dev0", False),
        ("not-a-version", True),
    ],
)
def test_resumability_tracks_the_installed_writer(monkeypatch, hf_version, resumable):
    """1.18 is the line: before it a partial is appended to, after it a new file is written."""
    monkeypatch.setattr("huggingface_hub.__version__", hf_version, raising = False)
    hf_cache_state.hf_partials_are_resumable.cache_clear()
    try:
        assert hf_cache_state.hf_partials_are_resumable() is resumable
    finally:
        hf_cache_state.hf_partials_are_resumable.cache_clear()


def test_unresumable_partial_is_purged_despite_a_matching_marker(monkeypatch, blobs):
    """The marker vouches for provenance, which is worth nothing with no resumer left."""
    monkeypatch.setattr(download_registry, "hf_partials_are_resumable", lambda: False)
    _prepare()  # writes the http marker
    partial = blobs / f"{_MAIN}.deadbeef.incomplete"
    partial.write_bytes(b"x" * 25)

    assert _prepare() == 1
    assert not partial.exists()


def test_resumable_partial_survives_a_matching_marker(monkeypatch, blobs):
    """Older hubs still append to it, so deleting it would throw away real bytes."""
    monkeypatch.setattr(download_registry, "hf_partials_are_resumable", lambda: True)
    _prepare()
    partial = blobs / f"{_MAIN}{hf_cache_state.INCOMPLETE_SUFFIX}"
    partial.write_bytes(b"x" * 25)

    assert _prepare() == 0
    assert partial.exists()


def test_a_peer_being_written_is_still_protected(monkeypatch, blobs):
    """Unresumable is not a licence to delete a blob another download is writing now."""
    monkeypatch.setattr(download_registry, "hf_partials_are_resumable", lambda: False)
    _prepare()
    mine = blobs / f"{_MAIN}.deadbeef.incomplete"
    mine.write_bytes(b"x" * 25)
    peer = blobs / f"{_PEER}.feedface.incomplete"
    peer.write_bytes(b"x" * 25)

    purged = download_registry.prepare_cache_for_transport(
        "model",
        "Org/Model",
        download_registry.TRANSPORT_HTTP,
        "Q4_K_M",
        only_blob_hashes = frozenset({_MAIN, _PEER}),
        protected_blob_hashes = frozenset({_PEER}),
    )

    assert purged == 1
    assert not mine.exists()
    assert peer.exists()
