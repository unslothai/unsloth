# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A user video load right after the Hub failed to connect: a cached base loads from the cache, not through
diffusers' untimed ``model_info``; anything missing retries online."""

from __future__ import annotations

import time
import types

import pytest

import utils.hf_xet_fallback as xet
from core.inference import diffusion as diffusion_mod
from core.inference import hub_reachability
from core.inference import video as video_mod
from core.inference.video import VideoBackend
from core.inference.video_families import detect_video_family

WAN_GGUF = "unsloth/Wan2.2-TI2V-5B-GGUF"
WAN_BASE = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_FILE = "wan2.2-ti2v-5b-Q4_K_M.gguf"


def _cache(root, repo, *names):
    folder = root / ("models--" + repo.replace("/", "--"))
    (folder / "refs").mkdir(parents = True, exist_ok = True)
    (folder / "refs" / "main").write_text("abc123")
    snap = folder / "snapshots" / "abc123"
    snap.mkdir(parents = True, exist_ok = True)
    for name in names:
        (snap / name).write_text("{}")


@pytest.fixture
def hub_down(monkeypatch, tmp_path):
    import huggingface_hub

    cache = tmp_path / "hub"
    cache.mkdir()
    monkeypatch.setattr(diffusion_mod, "hub_cache_dir", lambda: str(cache))
    monkeypatch.setattr(video_mod, "hub_cache_dir", lambda: str(cache))
    monkeypatch.setitem(
        hub_reachability._UNREACHABLE, hub_reachability.endpoint(), time.monotonic()
    )

    def _model_info(self, repo_id, **_kwargs):
        raise ConnectionError(f"model_info({repo_id!r}): the Hub did not connect")

    def _download(
        repo_id,
        filename,
        token = None,
        **kwargs,
    ):
        path = tmp_path / "dl" / filename
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_bytes(b"")
        return str(path)

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", _model_info, raising = False)
    monkeypatch.setattr(xet, "hf_hub_download_with_xet_fallback", _download)
    return cache


def _backend(monkeypatch, load_pipeline):
    backend = VideoBackend()
    backend._load_token = 1
    backend._loading = video_mod._VideoLoadingState(repo_id = WAN_GGUF, base_repo = WAN_BASE)
    fam = detect_video_family(WAN_BASE)
    monkeypatch.setattr(video_mod, "_detect_load_family", lambda *_a, **_k: fam)
    monkeypatch.setattr(backend, "load_pipeline", load_pipeline)
    return backend


def _cache_everything(cache):
    from core.inference.diffusion_families import mirror_repo
    for repo in {WAN_BASE, mirror_repo(WAN_BASE)}:
        _cache(cache, repo, "model_index.json")
    _cache(cache, WAN_GGUF, WAN_FILE)


def test_a_cached_base_loads_offline_while_the_hub_is_down(monkeypatch, hub_down):
    _cache_everything(hub_down)
    calls: list[dict] = []
    backend = _backend(monkeypatch, lambda **kw: calls.append(dict(kw)))

    backend._run_load(repo_id = WAN_GGUF, gguf_filename = WAN_FILE, _load_token = 1)

    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert [c.get("local_files_only") for c in calls] == [True]


def test_a_cache_miss_offline_retries_online(monkeypatch, hub_down):
    _cache_everything(hub_down)
    calls: list[dict] = []

    def _load(**kw):
        calls.append(dict(kw))
        if kw.get("local_files_only"):
            raise OSError("transformer/config.json is not in the cache")

    backend = _backend(monkeypatch, _load)
    backend._run_load(repo_id = WAN_GGUF, gguf_filename = WAN_FILE, _load_token = 1)

    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert [c.get("local_files_only") for c in calls] == [True, False]


def test_an_uncached_base_still_loads_online(monkeypatch, hub_down):
    _cache(hub_down, WAN_GGUF, WAN_FILE)
    calls: list[dict] = []
    backend = _backend(monkeypatch, lambda **kw: calls.append(dict(kw)))

    backend._run_load(repo_id = WAN_GGUF, gguf_filename = WAN_FILE, _load_token = 1)

    assert backend._loading is None, getattr(backend._loading, "error", None)
    assert [c.get("local_files_only") for c in calls] in ([False], [None])
