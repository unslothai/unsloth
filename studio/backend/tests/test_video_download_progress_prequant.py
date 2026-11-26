# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Video download progress counts hosted pre-cast DiT / text-encoder fetches made before the base pull."""

from __future__ import annotations

import inspect
import types

import pytest

from core.inference import video as video_mod
from core.inference.diffusion_prequant import PrequantSource
from core.inference.diffusion_te_prequant import TePrequantSource

BASE = "someone/fake-video-base"
PREQUANT = "someone/fake-video-prequant"
TE_KEY = "a1" * 32
DIT_KEY = "b2" * 32


def _sib(
    name,
    size,
    sha = None,
):
    return types.SimpleNamespace(
        rfilename = name,
        size = size,
        blob_id = "git-" + name,
        lfs = types.SimpleNamespace(sha256 = sha) if sha else None,
    )


class _FakeApi:
    def __init__(self, token = None):
        pass

    def model_info(
        self,
        repo,
        files_metadata = True,
        **_kw,
    ):
        if repo == PREQUANT:
            siblings = [
                _sib("text_encoder_fp8.safetensors", 400, TE_KEY),
                _sib("transformer_fp8.safetensors", 900, DIT_KEY),
            ]
        else:
            siblings = [
                _sib("model_index.json", 2),
                _sib("vae/config.json", 1),
                _sib("vae/diffusion_pytorch_model.safetensors", 30, "c3" * 32),
            ]
        return types.SimpleNamespace(sha = "0" * 40, siblings = siblings)


@pytest.fixture
def fake_cache(monkeypatch, tmp_path):
    monkeypatch.setattr("utils.hf_cache_settings.active_hf_hub_cache", lambda: str(tmp_path))
    monkeypatch.setattr(video_mod, "hub_cache_dir", lambda: str(tmp_path))
    import huggingface_hub
    import huggingface_hub.constants as hub_constants

    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "other"))
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_NO_MIRROR", raising = False)
    return tmp_path


def _partial(
    root,
    repo,
    key,
    nbytes,
    nonce = "1a2b3c4d",
):
    blobs = root / ("models--" + repo.replace("/", "--")) / "blobs"
    blobs.mkdir(parents = True, exist_ok = True)
    (blobs / f"{key}.{nonce}.incomplete").write_bytes(b"x" * nbytes)


def _start_load(backend, plan, expected):
    backend._loading = video_mod._VideoLoadingState(
        repo_id = BASE, base_repo = BASE, expected_bytes = expected
    )
    backend._loading.incomplete_baseline = video_mod.VideoBackend._plan_partials(plan)
    backend._loading.fetch_plan = plan


def _estimate(backend, **kwargs):
    planned: dict = {}
    expected = backend._estimate_download_bytes(
        BASE, None, BASE, None, "pipeline", planned_files = planned, **kwargs
    )
    return expected, planned


def test_te_prequant_fetch_is_counted_while_its_partial_grows(fake_cache):
    backend = video_mod.VideoBackend()
    te = TePrequantSource(kind = "repo", location = PREQUANT, filename = "text_encoder_fp8.safetensors")
    expected, plan = _estimate(backend, te_sources = {"text_encoder": te})
    assert expected == 2 + 1 + 30 + 400
    _start_load(backend, plan, expected)
    assert backend.load_progress()["downloaded_bytes"] == 0
    _partial(fake_cache, PREQUANT, TE_KEY, 150)
    assert backend.load_progress()["downloaded_bytes"] == 150


def test_denoiser_prequant_fetch_is_counted_while_its_partial_grows(fake_cache):
    backend = video_mod.VideoBackend()
    dit = PrequantSource(kind = "repo", location = PREQUANT, filename = "transformer_fp8.safetensors")
    expected, plan = _estimate(backend, denoiser_sources = [(dit, "fp8")])
    assert expected == 2 + 1 + 30 + 900
    _start_load(backend, plan, expected)
    _partial(fake_cache, PREQUANT, DIT_KEY, 600)
    assert backend.load_progress()["downloaded_bytes"] == 600


def test_a_mirrored_te_prequant_read_in_place_is_not_planned(fake_cache, monkeypatch):
    """A pre-cast encoder the loader reads in place from a mirror is never fetched, so it is not in the total."""
    monkeypatch.setattr(
        "core.inference.diffusion_te_prequant.te_prequant_mirror_path",
        lambda repo, name: str(fake_cache / name),
    )
    (fake_cache / "text_encoder_fp8.safetensors").write_bytes(b"m" * 400)
    backend = video_mod.VideoBackend()
    te = TePrequantSource(kind = "repo", location = PREQUANT, filename = "text_encoder_fp8.safetensors")
    expected, plan = _estimate(backend, te_sources = {"text_encoder": te})
    assert expected == 2 + 1 + 30
    assert all(entry.repo != PREQUANT for entry in plan.values())


def test_a_partial_present_at_plan_time_counts_its_growth(fake_cache):
    """A still-writing partial already on disk when the plan is made is counted from its baseline on."""
    backend = video_mod.VideoBackend()
    dit = PrequantSource(kind = "repo", location = PREQUANT, filename = "transformer_fp8.safetensors")
    expected, plan = _estimate(backend, denoiser_sources = [(dit, "fp8")])
    _partial(fake_cache, PREQUANT, DIT_KEY, 100, nonce = "0badf00d")
    _start_load(backend, plan, expected)
    assert backend.load_progress()["downloaded_bytes"] == 0
    _partial(fake_cache, PREQUANT, DIT_KEY, 250, nonce = "0badf00d")
    assert backend.load_progress()["downloaded_bytes"] == 150


def test_run_load_plans_the_same_prequant_sources_it_fetches():
    src = inspect.getsource(video_mod.VideoBackend._run_load)
    calls = src.split("self._estimate_download_bytes(")[1:]
    assert calls and all(
        "denoiser_sources = " in call.split("\n            )")[0] for call in calls
    )
    # The fetch reads the very list the plan was made from, not a second resolution.
    assert "self._fetch_denoiser_prequant(\n                    denoiser_sources," in src
