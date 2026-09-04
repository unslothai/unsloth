# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``media_locality`` reads the same verdict off both Hugging Face cache materialisations.

On Linux and macOS a snapshot entry is a relative symlink into ``blobs/``; on Windows without
developer mode ``huggingface_hub`` copies the blob into the snapshot instead. The completeness
check has to answer identically for the two, because the switch evicts the resident pipeline on
the strength of that answer: a component that reads as downloaded and then fails in
``from_pretrained`` leaves the user with nothing loaded.

The interesting state exists only in the symlink layout. A blob deleted by a cache sweep, or an
aborted pull, leaves the link behind, so a listing that matches on NAMES sees a complete
component where the copy layout would simply see an absent file.
"""

from __future__ import annotations

import os
from pathlib import Path

import core.inference.media_locality as locality
import pytest


def _cached_file(repo_dir: Path, snapshot: Path, name: str, payload: str, *, layout: str) -> Path:
    """One cached file, materialised the way *layout* says. Returns the blob path."""
    blob = repo_dir / "blobs" / f"blob-{name.replace('/', '-')}"
    blob.parent.mkdir(parents = True, exist_ok = True)
    blob.write_bytes(payload.encode("utf-8"))
    target = snapshot / name
    target.parent.mkdir(parents = True, exist_ok = True)
    if layout == "symlink":
        # the relative spelling huggingface_hub writes, so the tree survives being moved
        target.symlink_to(os.path.relpath(blob, target.parent))
    else:
        target.write_bytes(payload.encode("utf-8"))
    return blob


def _snapshot(tmp_path: Path, files: dict, *, layout: str) -> tuple[Path, Path]:
    repo_dir = tmp_path / "models--unsloth--z-image"
    snapshot = repo_dir / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    for name, payload in files.items():
        _cached_file(repo_dir, snapshot, name, payload, layout = layout)
    return repo_dir, snapshot


@pytest.mark.parametrize("layout", ["symlink", "copy"])
def test_a_complete_component_reads_the_same_in_both_cache_layouts(tmp_path, layout):
    _, snapshot = _snapshot(
        tmp_path,
        {
            "transformer/config.json": "{}",
            "transformer/diffusion_pytorch_model.safetensors": "weights",
        },
        layout = layout,
    )
    assert locality._component_present(snapshot / "transformer") is True


def test_a_dangling_weight_symlink_is_not_a_downloaded_component(tmp_path):
    """The blob is gone and the link remains: the copy layout cannot even express this."""
    repo_dir, snapshot = _snapshot(
        tmp_path,
        {
            "transformer/config.json": "{}",
            "transformer/diffusion_pytorch_model.safetensors": "weights",
        },
        layout = "symlink",
    )
    weight = snapshot / "transformer" / "diffusion_pytorch_model.safetensors"
    (repo_dir / "blobs" / "blob-transformer-diffusion_pytorch_model.safetensors").unlink()

    assert weight.is_symlink() and not weight.exists()
    assert locality._component_present(snapshot / "transformer") is False


def test_a_dangling_metadata_symlink_is_not_a_downloaded_component(tmp_path):
    """Same hole one branch over: a scheduler whose only config is a broken link."""
    repo_dir, snapshot = _snapshot(
        tmp_path, {"scheduler/scheduler_config.json": "{}"}, layout = "symlink"
    )
    (repo_dir / "blobs" / "blob-scheduler-scheduler_config.json").unlink()

    assert locality._component_present(snapshot / "scheduler") is False


def test_a_directory_named_like_a_weight_file_is_not_a_weight(tmp_path):
    """A name test alone would take the directory for the checkpoint it is named after."""
    _, snapshot = _snapshot(tmp_path, {"transformer/config.json": "{}"}, layout = "copy")
    (snapshot / "transformer" / "diffusion_pytorch_model.safetensors").mkdir()

    assert locality._component_present(snapshot / "transformer") is False


def test_a_pinned_variant_is_not_satisfied_by_a_dangling_link(tmp_path):
    """``variant`` names a weight set from_pretrained requires by name, so it needs a real file."""
    repo_dir, snapshot = _snapshot(
        tmp_path,
        {
            "transformer/config.json": "{}",
            "transformer/diffusion_pytorch_model.fp16.safetensors": "weights",
        },
        layout = "symlink",
    )
    (repo_dir / "blobs" / "blob-transformer-diffusion_pytorch_model.fp16.safetensors").unlink()

    assert locality._component_present(snapshot / "transformer", "fp16") is False
