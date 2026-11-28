# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A cold load of a retired FLUX.3 full copy is priced like the release it copies (CPU only).

``unsloth/flux-3-image-klein`` holds the whole BFL repo. On a 12 GB card its first load, before the transformer
bytes land, read "model size unknown; staying resident", kept everything on the card and ran out of memory in the
CUDA graph warm-up; the upstream id and its -FP8 mirror stream at the same budget from the family's bf16 table."""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from core.inference import diffusion as dmod
from core.inference.diffusion import DiffusionBackend
from core.inference.diffusion_families import _RETIRED_FULL_COPIES, detect_family, mirror_repo
from core.inference.diffusion_memory import OFFLOAD_NONE, DeviceMemory

REV = "0" * 40
MIB = 1024 * 1024
# The tester's simulated 12 GB card (ballast to 11904 MiB free, total capped at 12288 MiB), as the failing run logged.
FREE_MIB, TOTAL_MIB = 11_904, 12_288


def _safetensors(path: Path, mib: int) -> None:
    numel = mib * MIB // 2
    header = json.dumps(
        {"w": {"dtype": "BF16", "shape": [numel], "data_offsets": [0, numel * 2]}}
    ).encode()
    header += b" " * (-len(header) % 8)
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "wb") as fh:
        fh.write(len(header).to_bytes(8, "little") + header)
        fh.truncate(8 + len(header) + numel * 2)


def _partial_snapshot(hub: Path, repo_id: str, mib: int) -> None:
    """A FLUX.3 snapshot mid-download: manifest, text encoder and VAE landed, transformer not yet."""
    repo = hub / f"models--{repo_id.replace('/', '--')}"
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text(REV)
    snap = repo / "snapshots" / REV
    snap.mkdir(parents = True)
    (snap / "modular_model_index.json").write_text(
        json.dumps({"_class_name": "Flux3ModularPipeline"})
    )
    for folder in ("text_encoder", "vae"):
        (snap / folder).mkdir()
        (snap / folder / "config.json").write_text("{}")
        _safetensors(snap / folder / "model.safetensors", mib)


@pytest.fixture
def card12(monkeypatch, tmp_path):
    from huggingface_hub import constants as hf_constants

    live, other = tmp_path / "live-hub", tmp_path / "import-time-hub"
    live.mkdir()
    other.mkdir()
    monkeypatch.setattr(dmod, "hub_cache_dir", lambda: str(live))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(other))
    monkeypatch.setattr(
        dmod,
        "settled_snapshot_device_memory",
        lambda t, **_k: DeviceMemory("cuda", "cuda", "discrete_vram", FREE_MIB, TOTAL_MIB),
    )
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", supports_model_cpu_offload = True)
    return types.SimpleNamespace(target = target, hub = live)


def _plan(target, repo_id):
    fam = detect_family(repo_id)
    assert fam is not None
    # The load route passes the picked id as base and fetch repo (the retired copy has no mirror of its own).
    return DiffusionBackend()._plan_memory(
        target,
        None,
        repo_id,
        fam,
        None,
        False,
        kind = "pipeline",
        repo_id = repo_id,
        fetch_base = repo_id,
    )


_COPIES = sorted(_RETIRED_FULL_COPIES.items())


@pytest.mark.parametrize("copy,upstream", _COPIES)
@pytest.mark.parametrize("cached_mib", [0, 64])
def test_a_cold_retired_copy_is_not_planned_resident_at_12_gb(card12, copy, upstream, cached_mib):
    if cached_mib:
        _partial_snapshot(card12.hub, copy, cached_mib)
    plan = _plan(card12.target, copy)
    assert plan.estimates["model_dense_mib"] is not None, plan.reasons
    assert not any("model size unknown" in r for r in plan.reasons), plan.reasons
    assert plan.offload_policy != OFFLOAD_NONE, plan.reasons


@pytest.mark.parametrize("copy,upstream", _COPIES)
def test_a_cold_retired_copy_plans_as_its_release_and_mirror(card12, copy, upstream):
    """Same bytes, same plan: the copy, the upstream id and the -FP8 mirror price one release."""
    plans = {rid: _plan(card12.target, rid) for rid in (copy, upstream, mirror_repo(upstream))}
    want = plans[upstream]
    for rid, plan in plans.items():
        assert plan.offload_policy == want.offload_policy, (rid, plan.reasons)
        for key in ("model_dense_mib", "companion_dense_mib", "text_encoder_dense_mib"):
            assert plan.estimates[key] == want.estimates[key], (rid, key, plan.estimates)


def test_release_repo_id_maps_mirrors_and_retired_copies():
    for copy, upstream in _COPIES:
        assert dmod._release_repo_id(copy) == upstream
        assert dmod._release_repo_id(mirror_repo(upstream)) == upstream
        assert dmod._release_repo_id(upstream) == upstream
    assert (
        dmod._release_repo_id("someone/flux-3-image-klein-finetune")
        == "someone/flux-3-image-klein-finetune"
    )


def test_a_finetune_keeps_its_own_unknown_size(card12):
    """Only releases Studio knows the bytes of take the table; a fine-tune named like one does not."""
    fam = detect_family("black-forest-labs/flux-3-image-klein")
    assert (
        dmod._incomplete_release_cache_floor(
            fam, "someone/flux-3-image-klein-ft", False, None, 1, 1
        )
        is None
    )
