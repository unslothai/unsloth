# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Atomic ownership between STT and Model Hub writers."""

from __future__ import annotations

import pytest

from hub.utils.download_registry import DownloadRegistry


_CASES = (
    ("same-blob", "Org/Repo::Q4_K_M", "Q4_K_M", frozenset({"shared"})),
    ("disjoint-blob", "Org/Repo::Q8_0", "Q8_0", frozenset({"other"})),
    ("full-snapshot", "Org/Repo", None, frozenset({"snapshot"})),
    ("same-repo-variant", "Org/Repo::F16", "F16", frozenset({"variant"})),
)


@pytest.mark.parametrize("_label,key,variant,blob_hashes", _CASES)
def test_stt_owner_published_first_blocks_every_same_repo_model_hub_shape(
    _label, key, variant, blob_hashes
):
    registry = DownloadRegistry()
    owner = object()

    assert registry.claim_repository_owner("Org/Repo", owner) == (True, "owned")
    claimed, state = registry.claim(
        key,
        "xet",
        repo_type = "model",
        repo_id = "Org/Repo",
        variant = variant,
        blob_hashes = blob_hashes,
        progress_blob_hashes = blob_hashes,
    )

    assert claimed is False
    assert state == "repository_owned"
    assert registry.release_repository_owner("Org/Repo", owner) is True


@pytest.mark.parametrize("_label,key,variant,blob_hashes", _CASES)
def test_model_hub_published_first_blocks_stt_for_every_same_repo_shape(
    _label, key, variant, blob_hashes
):
    registry = DownloadRegistry()
    claimed, _state = registry.claim(
        key,
        "xet",
        repo_type = "model",
        repo_id = "Org/Repo",
        variant = variant,
        blob_hashes = blob_hashes,
        progress_blob_hashes = blob_hashes,
    )
    assert claimed is True

    assert registry.claim_repository_owner("Org/Repo", object())[0] is False


def test_stt_owner_and_model_hub_job_can_use_unrelated_repositories():
    registry = DownloadRegistry()
    owner = object()
    assert registry.claim_repository_owner("Org/Stt", owner)[0] is True

    claimed, _state = registry.claim(
        "Org/Chat::Q4_K_M",
        "xet",
        repo_type = "model",
        repo_id = "Org/Chat",
        variant = "Q4_K_M",
        blob_hashes = frozenset({"blob"}),
    )

    assert claimed is True
    assert registry.begin_delete("Org/Stt") is False
    assert registry.begin_delete("Org/Other") is True
    registry.end_delete("Org/Other")
    assert registry.release_repository_owner("Org/Stt", owner) is True
    assert registry.begin_delete("Org/Stt") is True


def test_only_the_current_stt_owner_can_release_a_repository():
    registry = DownloadRegistry()
    owner = object()
    assert registry.claim_repository_owner("Org/Repo", owner)[0] is True

    assert registry.release_repository_owner("Org/Repo", object()) is False
    assert registry.claim_repository_owner("Org/Repo", object())[0] is False
    assert registry.release_repository_owner("Org/Repo", owner) is True
