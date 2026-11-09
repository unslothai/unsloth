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


# Real CPU Inductor: a later model of the same classes runs earlier graphs from memory and records nothing.


class _Net(torch.nn.Module):
    def __init__(
        self,
        value: float,
        graph_break: bool = False,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((16, 16), value))
        self.graph_break = graph_break

    def forward(self, x):
        x = torch.sin(x @ self.weight)
        if self.graph_break:
            torch._dynamo.graph_break()
        return torch.cos(x) * 2


class _Other(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((16, 16), 0.5))

    def forward(self, x):
        return torch.tanh(x @ self.weight) + 1


@pytest.fixture
def real_compile(monkeypatch, tmp_path):
    monkeypatch.setenv(cc._ENV_MODE, "on")
    monkeypatch.setenv(cc._ENV_SYNC, "1")
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path / "bundles"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "inductor"))
    monkeypatch.delenv(ISOLATE_ENV, raising = False)
    monkeypatch.setattr(cc, "_live_dirs", {})
    monkeypatch.setattr(cc, "_collections", [], raising = False)
    monkeypatch.setattr(cc, "_collecting_since", None, raising = False)
    monkeypatch.setattr(cc, "_collecting_forwards", set(), raising = False)
    torch._dynamo.reset()
    try:
        with _cache.CacheArtifactManager.with_fresh_cache(), torch.no_grad():
            try:
                torch.compile(lambda x: x.sin() + 1, backend = "inductor")(torch.randn(2))
            except Exception as exc:  # noqa: BLE001 - no C++ toolchain for CPU Inductor here
                pytest.skip(f"CPU Inductor cannot compile here: {exc}")
            torch._dynamo.reset()
            _cache.CacheArtifactManager.clear()
            yield
    finally:
        torch._dynamo.reset()


def _compiled_lifecycle(
    name: str,
    net,
    shapes = ((4, 16),),
    reset_dynamo = False,
):
    ctx = cc.begin(
        family = name,
        transformer = net,
        dtype = "torch.float32",
        quant = None,
        attention_backend = "native",
        compile_kwargs = {"fullgraph": False, "dynamic": False},
    )
    assert ctx is not None
    if reset_dynamo:
        torch._dynamo.reset()
    compiled = torch.compile(net, dynamic = False)
    for shape in shapes:
        x = torch.randn(*shape)
        torch.testing.assert_close(compiled(x), net(x))
    saved = cc.save(ctx)
    cc.restore(ctx)
    return ctx, saved


def _inductor_keys(ctx) -> set:
    artifacts = _cache.CacheArtifactManager.deserialize(ctx.bundle.read_bytes())
    return {a.key for a in artifacts.get("inductor", [])}


def _unique_graphs() -> int:
    from torch._dynamo.utils import counters
    return counters["stats"]["unique_graphs"]


@pytest.mark.parametrize("graph_break", [False, True], ids = ["fullgraph", "graph_break"])
def test_a_model_reusing_an_earlier_models_compiled_graph_still_gets_a_bundle(
    real_compile, graph_break
):
    first, first_saved = _compiled_lifecycle("model_a", _Net(1.0, graph_break))
    graphs = _unique_graphs()
    second, second_saved = _compiled_lifecycle("model_b", _Net(2.0, graph_break))
    assert first.key != second.key
    # model_b compiled nothing: it ran model_a's graphs from Dynamo's in-memory cache
    assert _unique_graphs() == graphs
    assert first_saved and second_saved
    assert _inductor_keys(second) == _inductor_keys(first) != set()
    with _cache.CacheArtifactManager.with_fresh_cache():
        assert torch.compiler.load_cache_artifacts(second.bundle.read_bytes()) is not None


def test_an_unrelated_model_still_gets_only_its_own_artifacts(real_compile):
    first, _ = _compiled_lifecycle("model_a", _Net(1.0))
    other, saved = _compiled_lifecycle("other", _Other())
    assert saved
    assert _inductor_keys(other) and not _inductor_keys(other) & _inductor_keys(first)


def test_a_reload_that_compiles_a_new_shape_keeps_the_shape_it_reused(real_compile):
    net = _Net(1.0)
    first, _ = _compiled_lifecycle("model_a", net, shapes = ((4, 16),))
    reloaded, saved = _compiled_lifecycle("model_a", net, shapes = ((4, 16), (8, 16)))
    assert reloaded.key == first.key and saved
    keys = _inductor_keys(reloaded)
    assert _inductor_keys(first) < keys


def test_a_graph_recompiled_at_an_unchanged_entry_count_is_still_kept(real_compile):
    first, _ = _compiled_lifecycle("model_a", _Net(1.0))
    # reset + recompile at another shape: still one entry, but a new graph
    second, _ = _compiled_lifecycle("model_b", _Net(2.0), shapes = ((8, 16),), reset_dynamo = True)
    third, saved = _compiled_lifecycle("model_c", _Net(3.0), shapes = ((8, 16),))
    assert saved and _inductor_keys(second) - _inductor_keys(first)
    # model_a's graph left Dynamo, so its artifacts are not carried
    assert _inductor_keys(third) == _inductor_keys(second)
