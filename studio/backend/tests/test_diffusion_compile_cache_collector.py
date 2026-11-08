# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Each model's compile bundle holds its own artifacts, not every artifact torch collected since startup.

Real ``torch.compiler`` save / load and torch's real artifact collector, fed synthetic Inductor artifacts."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
_cache = pytest.importorskip("torch.compiler._cache")

from core.inference import diffusion_compile_cache as cc  # noqa: E402

ISOLATE_ENV = "UNSLOTH_DIFFUSION_COMPILE_CACHE_ISOLATE"
_PAYLOAD = 64 << 10


@pytest.fixture
def collector(monkeypatch, tmp_path):
    from torch._inductor.codecache import InductorCacheArtifact  # noqa: F401 - registers the type

    monkeypatch.setenv(cc._ENV_MODE, "on")
    monkeypatch.setenv(cc._ENV_SYNC, "1")
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    monkeypatch.delenv(ISOLATE_ENV, raising = False)
    monkeypatch.setattr(cc, "_live_dirs", {})
    # synthetic artifacts never populate a real inductor cache
    monkeypatch.setattr(_cache.CacheArtifact, "populate_cache", lambda self: None)
    with _cache.CacheArtifactManager.with_fresh_cache():
        yield


def _record(name: str) -> None:
    from torch._inductor.codecache import InductorCacheArtifact
    _cache.CacheArtifactManager.record_artifact(
        InductorCacheArtifact.type(), f"{name}-graph", bytes([ord(name[0])]) * _PAYLOAD
    )


def _lifecycle(name: str, restore: bool = True):
    ctx = cc.begin(
        transformer = types.SimpleNamespace(_repeated_blocks = [f"{name}Block"]),
        family = name,
        dtype = "torch.bfloat16",
        quant = None,
        attention_backend = "native",
        compile_kwargs = {"fullgraph": True},
    )
    assert ctx is not None and not ctx.hit
    _record(name)
    assert cc.save(ctx)
    if restore:
        cc.restore(ctx)
    return ctx


def _keys(ctx) -> set:
    artifacts = _cache.CacheArtifactManager.deserialize(ctx.bundle.read_bytes())
    return {a.key for group in artifacts.values() for a in group}


def test_each_models_bundle_holds_only_its_own_artifacts(collector):
    ctxs = [_lifecycle(name) for name in ("alpha", "beta", "gamma", "delta")]
    sizes = [c.bundle.stat().st_size for c in ctxs]
    assert (
        max(sizes) - min(sizes) < 1024
    )  # 1 artifact each; torch's collector alone gives 1, 2, 3, 4
    assert [_keys(c) for c in ctxs] == [{f"{n}-graph"} for n in ("alpha", "beta", "gamma", "delta")]


def test_a_saved_bundle_still_loads_into_a_fresh_collector(collector):
    first = _lifecycle("alpha")
    _lifecycle("beta")
    with _cache.CacheArtifactManager.with_fresh_cache():
        assert torch.compiler.load_cache_artifacts(first.bundle.read_bytes()) is not None


def test_the_isolation_switch_keeps_torchs_accumulating_collector(collector, monkeypatch):
    monkeypatch.setenv(ISOLATE_ENV, "0")
    assert cc._ENV_ISOLATE == ISOLATE_ENV
    ctxs = [_lifecycle(name) for name in ("alpha", "beta", "gamma")]
    sizes = [c.bundle.stat().st_size for c in ctxs]
    assert sizes[1] > sizes[0] + _PAYLOAD // 2 and sizes[2] > sizes[1] + _PAYLOAD // 2
    assert _keys(ctxs[2]) == {"alpha-graph", "beta-graph", "gamma-graph"}


def test_a_bundle_still_live_keeps_the_collector(collector):
    # a reload overlapping its predecessor's teardown: the live model's pending artifacts are not dropped
    first = _lifecycle("alpha", restore = False)
    second = _lifecycle("beta")
    assert _keys(second) == {"alpha-graph", "beta-graph"}
    cc.restore(first)


def test_a_running_compile_is_never_waited_for(collector, monkeypatch):
    import threading

    lock = threading.Lock()
    lock.acquire()
    monkeypatch.setattr(cc, "_compile_lock", lambda: lock)
    _record("pending")
    assert cc._fresh_collector(None) is False
    assert _cache.CacheArtifactManager.need_serialize()
    lock.release()
    assert cc._fresh_collector(None) is True
    assert not _cache.CacheArtifactManager.need_serialize()
