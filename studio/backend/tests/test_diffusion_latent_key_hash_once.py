# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The latent cache reads each image's bytes once per pass, not once per crop/flip variant."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import torch
from PIL import Image

from core.training import diffusion_dit_trainer as dit
from core.training import diffusion_train_extras as extras
from core.training.diffusion_train_extras import PersistentConditioningCache

_VARIANTS = [(0.1, 0.2, False), (0.3, 0.4, True), (0.5, 0.6, False), (0.7, 0.8, True)]


def _warm_cache(tmp_path, n = 3):
    paths = []
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (64, 64), (i * 40, 10, 20)).save(p)
        paths.append(str(p))
    cache = PersistentConditioningCache(str(tmp_path / "cc"), "flux.1", 512)
    plan = [list(_VARIANTS) for _ in paths]
    for p in paths:
        for v in _VARIANTS:
            cache.put(cache.latent_key(p, v), (torch.zeros(1, 4, 8, 8), None))
    return cache, paths, plan


def _count_hashes(monkeypatch):
    seen = Counter()
    real = extras._file_content_hash

    def counting(path):
        seen[path] += 1
        return real(path)

    monkeypatch.setattr(extras, "_file_content_hash", counting)
    return seen


def test_warm_conditioning_hashes_each_image_once(tmp_path, monkeypatch):
    cache, paths, plan = _warm_cache(tmp_path)
    seen = _count_hashes(monkeypatch)
    _, latents = dit._load_warm_conditioning(cache, paths, plan, [], "cpu")
    assert latents is not None and [len(v) for v in latents] == [4] * len(paths)
    assert seen == Counter({p: 1 for p in paths})


def test_build_latent_cache_hashes_each_image_once(tmp_path, monkeypatch):
    cache, paths, plan = _warm_cache(tmp_path)
    seen = _count_hashes(monkeypatch)
    cfg = SimpleNamespace(resolution = 512, center_crop = False)
    out = dit._build_latent_cache(
        None, None, paths, cfg, "cpu", torch.float32, None, lambda: False, pcache = cache, plan = plan
    )
    assert [len(v) for v in out] == [4] * len(paths)
    assert seen == Counter({p: 1 for p in paths})


def test_shared_hash_keeps_the_key_and_still_tracks_edits(tmp_path):
    cache, paths, _ = _warm_cache(tmp_path, n = 1)
    path = paths[0]
    for shade, shape in ((200, None), (250, (448, 576))):
        digest = cache.content_hash(path)
        before = [cache.latent_key(path, v, shape, content_hash = digest) for v in _VARIANTS]
        assert before == [cache.latent_key(path, v, shape) for v in _VARIANTS]
        Image.new("RGB", (64, 64), (shade, shade, shade)).save(path)
        after = [
            cache.latent_key(path, v, shape, content_hash = cache.content_hash(path))
            for v in _VARIANTS
        ]
        assert all(a != b for a, b in zip(after, before))
