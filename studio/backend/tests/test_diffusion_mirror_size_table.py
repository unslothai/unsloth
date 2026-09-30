# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An uncached unsloth mirror is sized from its family's bf16 table, like its upstream (empty caches, no GPU)."""

from __future__ import annotations

import types

import pytest

from core.inference import diffusion as dmod
from core.inference.diffusion import DiffusionBackend
from core.inference.diffusion_families import canonical_base, detect_family
from core.inference.diffusion_memory import OFFLOAD_NONE, DeviceMemory

MIRROR = "unsloth/FLUX.1-schnell"
UPSTREAM = "black-forest-labs/FLUX.1-schnell"


@pytest.fixture
def l4_with_empty_cache(monkeypatch, tmp_path):
    from huggingface_hub import constants as hf_constants

    live, other = tmp_path / "live-hub", tmp_path / "import-time-hub"
    live.mkdir()
    other.mkdir()
    monkeypatch.setattr(dmod, "hub_cache_dir", lambda: str(live))
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(other))
    # A 24 GB L4 with ~22 GB free.
    monkeypatch.setattr(
        dmod,
        "settled_snapshot_device_memory",
        lambda t: DeviceMemory("cuda", "cuda", "discrete_vram", 22_000, 23_034),
    )
    monkeypatch.setattr(dmod, "estimate_image_runtime_mib", lambda **kw: 1_000)
    return types.SimpleNamespace(device = "cuda", backend = "cuda", supports_model_cpu_offload = True)


def _plan(target, repo_id):
    fam = detect_family(repo_id)
    assert fam is not None and fam.base_repo.lower() == UPSTREAM.lower()
    return DiffusionBackend()._plan_memory(
        target, None, repo_id, fam, None, False, kind = "pipeline", repo_id = repo_id
    )


def test_the_mirror_table_pairs_this_repo_with_the_family_base():
    assert canonical_base(MIRROR) == UPSTREAM


def test_an_uncached_mirror_is_sized_from_the_family_table(l4_with_empty_cache):
    plan = _plan(l4_with_empty_cache, MIRROR)
    assert plan.estimates["model_dense_mib"] is not None, plan.reasons
    assert plan.offload_policy != OFFLOAD_NONE, plan.reasons
    assert not any("unknown" in reason for reason in plan.reasons), plan.reasons


def test_the_mirror_plans_exactly_like_its_upstream(l4_with_empty_cache):
    mirror, upstream = _plan(l4_with_empty_cache, MIRROR), _plan(l4_with_empty_cache, UPSTREAM)
    assert mirror.offload_policy == upstream.offload_policy
    for key in ("model_dense_mib", "companion_dense_mib", "text_encoder_dense_mib"):
        assert mirror.estimates[key] == upstream.estimates[key], key


def test_an_unrelated_repo_still_sizes_from_its_own_bytes(l4_with_empty_cache):
    """Only a known mirror borrows the table: a fine-tune in the same family, uncached, stays
    unknown exactly as before."""
    fam = detect_family(UPSTREAM)
    plan = DiffusionBackend()._plan_memory(
        l4_with_empty_cache,
        None,
        "someone/flux-schnell-finetune",
        fam,
        None,
        False,
        kind = "pipeline",
        repo_id = "someone/flux-schnell-finetune",
    )
    assert plan.estimates["model_dense_mib"] is None
