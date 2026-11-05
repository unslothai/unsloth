# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Offline loads must use the mirror the online prefetch staged, not an upstream snapshot holding only a stray file."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.inference import diffusion as D
from core.inference.diffusion import DiffusionBackend, _LoadingState
from utils import hf_cache_settings


def _snapshot(hub: Path, repo_id: str, sha: str, files: dict[str, bytes]) -> Path:
    repo = hub / ("models--" + repo_id.replace("/", "--"))
    (repo / "refs").mkdir(parents = True, exist_ok = True)
    (repo / "refs" / "main").write_text(sha)
    snap = repo / "snapshots" / sha
    for name, data in files.items():
        (snap / name).parent.mkdir(parents = True, exist_ok = True)
        (snap / name).write_bytes(data)
    return snap


_PIPELINE = {
    "model_index.json": b"{}",
    "unet/config.json": b"{}",
    "unet/diffusion_pytorch_model.safetensors": b"0",
    "vae/config.json": b"{}",
    "vae/diffusion_pytorch_model.safetensors": b"0",
}

# (upstream, mirror, what the stray download left in the upstream snapshot)
_CASES = [
    (
        "stabilityai/stable-diffusion-xl-base-1.0",
        "unsloth/stable-diffusion-xl-base-1.0",
        {"sd_xl_base_1.0.safetensors": b"0"},
    ),
    (
        "Tongyi-MAI/Z-Image-Turbo",
        "unsloth/Z-Image-Turbo",
        {
            "text_encoder/model-00001-of-00003.safetensors": b"0",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": b"0",
        },
    ),
]


@pytest.fixture
def hub(monkeypatch, tmp_path):
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.delenv("UNSLOTH_DIFFUSION_NO_MIRROR", raising = False)
    monkeypatch.setattr(hf_cache_settings, "active_hf_hub_cache", lambda: str(hub))
    monkeypatch.setattr(D, "hub_cache_dir", lambda: str(hub))
    return hub


def _offline_fetch_base(monkeypatch, repo_id: str) -> str:
    loaded: dict = {}
    monkeypatch.setattr(DiffusionBackend, "_prefetch_files", lambda self, *a, **k: None)
    monkeypatch.setattr(DiffusionBackend, "load_pipeline", lambda self, **k: loaded.update(k))
    backend = DiffusionBackend()
    backend._loading = _LoadingState(repo_id = repo_id, base_repo = repo_id)
    backend._run_load(
        repo_id = repo_id,
        local_files_only = True,
        _load_token = backend._load_token,
    )
    assert backend._loading is None, "the staging phase raised before load_pipeline"
    assert loaded.get("local_files_only") is True
    return loaded["_fetch_base"]


@pytest.mark.parametrize("upstream,mirror,stray", _CASES)
def test_offline_load_reads_the_mirror_prefetch_staged(monkeypatch, hub, upstream, mirror, stray):
    _snapshot(hub, upstream, "a" * 40, stray)
    _snapshot(hub, mirror, "b" * 40, _PIPELINE)
    assert _offline_fetch_base(monkeypatch, upstream) == mirror


@pytest.mark.parametrize("upstream,mirror,stray", _CASES)
def test_offline_load_keeps_a_full_upstream_pipeline(monkeypatch, hub, upstream, mirror, stray):
    _snapshot(hub, upstream, "a" * 40, {**stray, **_PIPELINE})
    _snapshot(hub, mirror, "b" * 40, _PIPELINE)
    assert _offline_fetch_base(monkeypatch, upstream) == upstream


def test_without_a_cached_mirror_manifest_the_old_decision_stands(hub):
    upstream, mirror, stray = _CASES[0]
    _snapshot(hub, upstream, "a" * 40, stray)
    assert D._cached_mirror_manifest(upstream, "model_index.json") == ()
    _snapshot(hub, mirror, "b" * 40, _PIPELINE)
    assert D._cached_mirror_manifest(upstream, "model_index.json") == ("model_index.json",)
    assert D._cached_mirror_manifest("someone/finetune", "model_index.json") == ()
