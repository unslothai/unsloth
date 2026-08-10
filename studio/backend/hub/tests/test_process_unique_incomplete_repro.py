# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused reproduction for Hugging Face's process-unique partial filenames."""

from types import SimpleNamespace

from hub.services import snapshot_progress


def test_progress_counts_process_unique_incomplete_blob(monkeypatch, tmp_path):
    """An active ``<etag>.<uuid>.incomplete`` target must contribute bytes."""
    entry = tmp_path / "models--Org--Model-GGUF"
    blobs = entry / "blobs"
    blobs.mkdir(parents = True)
    (blobs / "mainhash.deadbeef.incomplete").write_bytes(b"x" * 5)

    monkeypatch.setattr(
        snapshot_progress,
        "preferred_repo_cache_dirs",
        lambda *_args, **_kwargs: [entry],
    )
    registry = SimpleNamespace(
        get_job = lambda _key: SimpleNamespace(state = "running"),
        get_job_metadata = lambda _key: SimpleNamespace(completed_baseline_bytes = 0),
    )

    result = snapshot_progress.compute_snapshot_progress(
        repo_type = "model",
        repo_id = "Org/Model-GGUF",
        job_key = "model:org/model-gguf#q4_k_m",
        expected_bytes = 100,
        hf_token = None,
        registry = registry,
        metadata_resolver = lambda *_args: (100, frozenset({"mainhash"})),
        variant = "Q4_K_M",
    )

    assert result["completed_bytes"] == 0
    assert result["downloaded_bytes"] == 5
    assert result["progress"] == 0.05
