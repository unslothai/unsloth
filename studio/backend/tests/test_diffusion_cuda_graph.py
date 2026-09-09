# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the denoiser CUDA-graph layer (``diffusion_cuda_graph.py``).

Hermetic by default: the ``stub_torch`` fixture installs a fake torch into ``sys.modules``, so
every refusal, the cap, poisoning and the lifecycle run without a GPU. The last test imports real
torch inside the body, skips without CUDA, and asserts bit-identity.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import sys
import types
import weakref

import pytest

from core.inference import diffusion_cuda_graph as cg


def _contiguous_stride(shape) -> tuple:
    stride: list = []
    acc = 1
    for dim in reversed(shape):
        stride.append(acc)
        acc *= dim
    return tuple(reversed(stride))


class _FakeTensor:
    """Everything the layer touches on a tensor: shape, stride(), dtype, device, copy_, clone."""

    def __init__(
        self,
        shape = (1, 4),
        *,
        dtype = "bfloat16",
        device_type = "cuda",
        device_index = 0,
        stride = None,
        value = None,
        tag = "in",
    ):
        self.shape = tuple(shape)
        self._stride = tuple(stride) if stride is not None else _contiguous_stride(self.shape)
        self.dtype = dtype
        self.device = types.SimpleNamespace(type = device_type, index = device_index)
        self.value = value
        self.tag = tag
        self.copied_from: list = []
        self.clone_of = None

    def stride(self):
        return self._stride

    def copy_(self, src):
        self.copied_from.append(src)
        self.value = src.value
        return self

    def clone(self):
        out = _FakeTensor(
            self.shape,
            dtype = self.dtype,
            device_type = self.device.type,
            device_index = self.device.index,
            stride = self._stride,
            value = self.value,
            tag = self.tag,
        )
        out.clone_of = self
        return out

    def __repr__(self):  # pragma: no cover - debugging aid only
        return f"_FakeTensor({self.shape}, {self.tag}, value = {self.value})"


class _FakeGraph:
    def __init__(self):
        self.replays = 0
        self._pool = ("pool", id(self))

    def replay(self):
        self.replays += 1

    def pool(self):
        return self._pool


class _FakeStream:
    def __init__(self):
        self.waited: list = []

    def wait_stream(self, other):
        self.waited.append(other)


def _build_stub_torch():
    """Just enough torch for the capture path, plus a record of what the layer did to it."""
    torch = types.ModuleType("torch")
    records = {
        "graphs": [],  # (graph, pool) per torch.cuda.graph(...)
        "streams": [],
        "synchronize": 0,
        "empty_cache": 0,
        "inference_mode": [],
        "graph_error": None,  # set to an exception to make CUDAGraph() raise
    }
    torch._records = records

    torch.is_tensor = lambda obj: isinstance(obj, _FakeTensor)

    def _empty_like(tensor):
        return _FakeTensor(
            tensor.shape,
            dtype = tensor.dtype,
            device_type = tensor.device.type,
            device_index = tensor.device.index,
            stride = tensor.stride(),
            tag = "static",
        )

    torch.empty_like = _empty_like

    @contextlib.contextmanager
    def _inference_mode(mode = True):
        records["inference_mode"].append(mode)
        yield

    torch.inference_mode = _inference_mode

    def _cuda_graph():
        if records["graph_error"] is not None:
            raise records["graph_error"]
        return _FakeGraph()

    @contextlib.contextmanager
    def _graph(graph, pool = None):
        records["graphs"].append((graph, pool))
        yield

    @contextlib.contextmanager
    def _stream(stream):
        records["streams"].append(stream)
        yield

    def _synchronize(*args, **kwargs):
        records["synchronize"] += 1

    def _empty_cache():
        records["empty_cache"] += 1

    current = _FakeStream()
    torch.cuda = types.SimpleNamespace(
        CUDAGraph = _cuda_graph,
        graph = _graph,
        Stream = _FakeStream,
        stream = _stream,
        current_stream = lambda: current,
        synchronize = _synchronize,
        empty_cache = _empty_cache,
        is_available = lambda: True,
    )
    return torch


class _FakeDiT:
    """Stand-in for a denoiser module: a class-level ``forward`` and an instance ``__dict__``."""

    def __init__(self):
        self.calls = 0
        self.seen: list = []

    def forward(
        self,
        hidden_states,
        timestep = None,
        return_dict = True,
    ):
        self.calls += 1
        self.seen.append((hidden_states, timestep, return_dict))
        return (_FakeTensor((1, 4), value = ("out", self.calls), tag = "out"),)


class UNet2DConditionModel:  # noqa: N801 - the class NAME is what _denoiser_unet gates on
    pass


@pytest.fixture
def stub_torch(monkeypatch):
    torch = _build_stub_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


@pytest.fixture(autouse = True)
def _clean_module_globals():
    """The shared pool and the live-wrapper set are process-wide, so no test may inherit them."""
    cg._POOL_BOX[0] = None
    cg._LIVE_WRAPPERS.clear()
    yield
    cg._POOL_BOX[0] = None
    cg._LIVE_WRAPPERS.clear()


def _t(shape = (1, 4), **kwargs):
    return _FakeTensor(shape, **kwargs)


def _armed(module = None, **kwargs):
    handle = cg.GraphedForward(module if module is not None else _FakeDiT(), **kwargs)
    return handle.enable()


def test_flatten_rebuild_round_trip(stub_torch):
    a, b, c = _t(), _t((2, 2)), _t((3,))
    tree = ((a, [b, c]), {"guidance": None, "shapes": (1, 2), "attn": {"scale": 0.5}})
    out: list = []
    spec = cg._flatten(tree, out)
    assert out == [a, b, c]
    rebuilt = cg._rebuild(spec, out)
    assert rebuilt == tree


def test_flatten_keeps_tuple_and_list_apart(stub_torch):
    out: list = []
    spec = cg._flatten([_t(), (_t(),)], out)
    rebuilt = cg._rebuild(spec, out)
    assert isinstance(rebuilt, list)
    assert isinstance(rebuilt[1], tuple)


def test_flatten_refuses_unknown_object_and_keys_it_as_o(stub_torch):
    class Weird:
        pass

    weird = Weird()
    with pytest.raises(TypeError):
        cg._flatten({"controlnet": weird}, [])
    key = cg.graph_key(weird)
    assert key[0] == "o" and key[1] == "Weird"
    assert cg._uncapturable(cg.graph_key({"controlnet": weird})) is True
    assert cg._uncapturable(cg.graph_key({"controlnet": _t()})) is False


def test_graph_key_distinguishes_metadata(stub_torch):
    base = cg.graph_key(_t((2, 4)))
    assert base == cg.graph_key(_t((2, 4)))  # equal metadata -> equal key -> one graph
    assert base != cg.graph_key(_t((2, 8)))
    assert base != cg.graph_key(_t((2, 4), stride = (1, 2)))
    assert base != cg.graph_key(_t((2, 4), dtype = "float16"))
    assert base != cg.graph_key(_t((2, 4), device_type = "cpu"))
    assert base != cg.graph_key(_t((2, 4), device_index = 1))


def test_has_float_only_for_real_floats(stub_torch):
    assert cg._has_float(cg.graph_key({"timestep": 0.5})) is True
    assert cg._has_float(cg.graph_key([1, True, "x", None, _t()])) is False
    assert cg._has_float(cg.graph_key((1, [2, {"a": 3.0}]))) is True


def test_return_dict_true_or_absent_runs_eager(stub_torch):
    module = _FakeDiT()
    handle = _armed(module)
    handle(_t(), timestep = _t((1,)))  # absent -> defaults to True
    handle(_t(), timestep = _t((1,)), return_dict = True)
    assert handle.stats["eager_calls"] == 2
    assert handle.stats["captures"] == 0
    assert module.calls == 2


def test_capture_then_replay_uses_one_graph_and_copies_statics(stub_torch):
    module = _FakeDiT()
    handle = _armed(module)
    first, second = _t(value = 1.0), _t(value = 2.0)
    handle(first, timestep = _t((1,), value = 10.0), return_dict = False)
    handle(second, timestep = _t((1,), value = 20.0), return_dict = False)

    assert handle.stats["captures"] == 1
    assert handle.stats["replays"] == 2
    assert len(handle.cache) == 1
    entry = next(iter(handle.cache.values()))
    # One copy at capture, one per call: the live values reach the static buffers every time.
    assert entry.static[0].copied_from == [first, first, second]
    assert entry.static[0].value == 2.0
    assert entry.graph.replays == 2
    # Warm-ups plus the capture call, all on the ORIGINAL forward.
    assert module.calls == cg.WARMUP_ITERS + 1
    assert stub_torch._records["synchronize"] == 1
    assert stub_torch._records["inference_mode"] == [False]


def test_new_shape_adds_a_second_graph(stub_torch):
    handle = _armed()
    handle(_t((1, 4)), timestep = _t((1,)), return_dict = False)
    handle(_t((2, 4)), timestep = _t((1,)), return_dict = False)
    assert handle.stats["captures"] == 2
    assert len(handle.cache) == 2
    assert handle.poisoned is False


def test_outputs_are_cloned_out_of_the_pool(stub_torch):
    handle = _armed()
    out = handle(_t(), timestep = _t((1,)), return_dict = False)
    entry = next(iter(handle.cache.values()))
    assert out[0] is not entry.out_tensors[0]
    assert out[0].clone_of is entry.out_tensors[0]


def test_float_leaf_refuses_without_capturing(stub_torch):
    module = _FakeDiT()
    handle = _armed(module)
    handle(_t(), timestep = 0.5, return_dict = False)
    assert handle.stats["refused_float"] == 1
    assert handle.stats["captures"] == 0
    assert handle.stats["eager_calls"] == 1
    assert handle.poisoned is False
    assert module.calls == 1


def test_unknown_object_in_the_tree_refuses_without_capturing(stub_torch):
    class Weird:
        pass

    handle = _armed()
    handle(_t(), timestep = Weird(), return_dict = False)
    assert handle.stats["refused_object"] == 1
    assert handle.stats["captures"] == 0
    assert handle.poisoned is False


def test_host_tensor_poisons_with_capture_error(stub_torch):
    module = _FakeDiT()
    handle = _armed(module)
    out = handle(_t(device_type = "cpu"), timestep = _t((1,)), return_dict = False)

    assert handle.poisoned is True
    assert handle.stats["refused_host_tensor"] == 1
    assert handle.stats["fallbacks"] == 1
    assert handle.capture_error["type"] == "RuntimeError"
    assert "not on cuda" in handle.capture_error["msg"]
    assert handle.capture_error["traceback"]
    assert module.calls == 1  # eager, and nothing was warmed up first
    assert out[0].value == ("out", 1)
    # Poisoned means eager forever, with no further capture attempts.
    handle(_t(), timestep = _t((1,)), return_dict = False)
    assert handle.stats["captures"] == 0
    assert module.calls == 2


def test_capture_exception_poisons_and_returns_the_eager_result(stub_torch):
    stub_torch._records["graph_error"] = RuntimeError("CUDA out of memory during capture")
    module = _FakeDiT()
    logged: list = []
    logger = types.SimpleNamespace(
        warning = lambda *a: logged.append(a),
        info = lambda *a: None,
        debug = lambda *a: None,
    )
    handle = cg.GraphedForward(module, logger = logger).enable()
    out = handle(_t(), timestep = _t((1,)), return_dict = False)

    assert handle.poisoned is True
    assert handle.capture_error["type"] == "RuntimeError"
    assert handle.stats["fallbacks"] == 1
    assert module.calls == cg.WARMUP_ITERS + 1  # warm-ups ran, then CUDAGraph() raised
    assert out[0].value == ("out", module.calls)
    assert len(logged) == 1
    assert "%s" in logged[0][0] and "capture failed" in logged[0][0]
    # Type and message only in the log line; the traceback stays on capture_error.
    assert all("Traceback" not in str(part) for part in logged[0][1:])


def test_a_failed_capture_is_released_before_the_eager_fallback(stub_torch):
    """The handled exception owns _capture's frame and so its statics; the eager fallback must run
    after they are dropped, or an OOM capture turns into an OOM render."""
    statics: list = []
    plain_empty_like = stub_torch.empty_like

    def _recording_empty_like(tensor):
        out = plain_empty_like(tensor)
        statics.append(weakref.ref(out))
        return out

    stub_torch.empty_like = _recording_empty_like
    stub_torch._records["graph_error"] = RuntimeError("CUDA out of memory during capture")

    seen: list = []

    class _Probe(_FakeDiT):
        def forward(
            self,
            hidden_states,
            timestep = None,
            return_dict = True,
        ):
            self.calls += 1
            seen.append(
                {
                    "live_statics": sum(1 for ref in statics if ref() is not None),
                    "empty_cache": stub_torch._records["empty_cache"],
                }
            )
            return (_FakeTensor((1, 4), value = ("out", self.calls), tag = "out"),)

    handle = _armed(_Probe())
    out = handle(_t(), timestep = _t((1,)), return_dict = False)

    assert handle.poisoned is True
    assert out[0].value == ("out", cg.WARMUP_ITERS + 1)
    assert statics
    assert seen[0]["live_statics"] == len(statics)
    assert seen[-1]["live_statics"] == 0
    assert seen[-1]["empty_cache"] == 1
    assert handle.capture_error["traceback"]
    assert handle.capture_error["type"] == "RuntimeError"


def test_poisoning_drops_the_graphs_it_can_no_longer_replay(stub_torch):
    """``__call__`` short-circuits on ``poisoned`` before it reads the cache, so an entry captured
    before the failure can never be replayed again: keeping it pins its statics, its outputs and its
    slice of the pool for the life of the load, against the eager fallback the poisoning falls back
    TO. Measured with the real wrapper: a 512-shape graph held through a failed larger capture is the
    difference between that render OOMing and completing."""
    module = _FakeDiT()
    handle = _armed(module)
    handle(_t((1, 4)), timestep = _t((1,)), return_dict = False)
    assert len(handle.cache) == 1
    assert cg._POOL_BOX[0] is not None

    # A second shape fails to capture.
    stub_torch._records["graph_error"] = RuntimeError("CUDA out of memory during capture")
    out = handle(_t((2, 4)), timestep = _t((1,)), return_dict = False)

    assert handle.poisoned is True
    assert out[0].value[0] == "out"  # the eager result still came back
    assert handle.cache == {}, "graphs that can never replay again are still pinning memory"
    assert cg._POOL_BOX[0] is None, "and the pool token outlived the last graph in it"
    # Still eager from here on, and it does not try to capture again.
    before = module.calls
    handle(_t((1, 4)), timestep = _t((1,)), return_dict = False)
    assert module.calls == before + 1
    assert handle.stats["captures"] == 1


def test_graph_cap_degrades_to_eager_without_poisoning(stub_torch):
    module = _FakeDiT()
    logged: list = []
    logger = types.SimpleNamespace(
        warning = lambda *a: logged.append(a),
        info = lambda *a: None,
        debug = lambda *a: None,
    )
    handle = cg.GraphedForward(module, max_graphs = 2, logger = logger).enable()
    for width in (4, 8, 16, 32):
        handle(_t((1, width)), timestep = _t((1,)), return_dict = False)

    assert handle.stats["captures"] == 2
    assert handle.stats["cap_skips"] == 2
    assert handle.poisoned is False
    assert len(handle.cache) == 2
    assert len(logged) == 1  # one warning, not one per skipped shape

    # The graphs already recorded keep replaying.
    before = handle.stats["replays"]
    handle(_t((1, 4)), timestep = _t((1,)), return_dict = False)
    assert handle.stats["replays"] == before + 1
    assert handle.stats["captures"] == 2


def test_bypass_runs_eager_and_keeps_the_graphs(stub_torch):
    module = _FakeDiT()
    handle = _armed(module)
    handle(_t(), timestep = _t((1,)), return_dict = False)
    assert len(handle.cache) == 1

    cg.set_bypass([handle], True)
    handle(_t(), timestep = _t((1,)), return_dict = False)
    assert handle.bypassed is True
    assert handle.stats["eager_calls"] == 1
    assert handle.stats["replays"] == 1
    assert len(handle.cache) == 1  # graphs survive the bypass

    cg.set_bypass([handle], False)
    handle(_t(), timestep = _t((1,)), return_dict = False)
    assert handle.stats["replays"] == 2
    assert handle.stats["captures"] == 1


def test_step_cache_marker_forces_eager(stub_torch):
    module = _FakeDiT()
    handle = _armed(module)
    module._unsloth_step_cache = object()
    handle(_t(), timestep = _t((1,)), return_dict = False)
    assert handle.stats["eager_calls"] == 1
    assert handle.stats["captures"] == 0

    module._unsloth_step_cache = None
    handle(_t(), timestep = _t((1,)), return_dict = False)
    assert handle.stats["captures"] == 1


def test_disabled_wrapper_runs_eager_and_uninstalls(stub_torch):
    module = _FakeDiT()
    handle = _armed(module)
    handle.disable()
    assert "forward" not in module.__dict__
    handle(_t(), timestep = _t((1,)), return_dict = False)
    assert handle.stats["eager_calls"] == 1


def test_reset_drops_the_graphs(stub_torch):
    handle = _armed()
    handle(_t(), timestep = _t((1,)), return_dict = False)
    cg.reset_all([handle])
    assert handle.cache == {}
    assert stub_torch._records["empty_cache"] == 1

    handle(_t(), timestep = _t((1,)), return_dict = False)
    assert handle.stats["captures"] == 2  # a LoRA swap re-captures rather than serving old weights


def test_install_uninstall_restores_the_class_forward(stub_torch):
    module = _FakeDiT()
    pipe = types.SimpleNamespace(transformer = module)
    handles = cg.install_cuda_graphs(pipe)

    assert len(handles) == 1
    assert pipe._unsloth_cuda_graphs == handles
    assert module.__dict__["forward"] is handles[0]
    assert module.forward is handles[0]

    handles[0](_t(), timestep = _t((1,)), return_dict = False)
    cg.uninstall_all(handles)

    assert "forward" not in module.__dict__
    assert module.forward.__func__ is _FakeDiT.forward
    assert handles[0].cache == {}
    cg.uninstall_all(handles)  # idempotent
    cg.uninstall_all(())
    cg.uninstall_all(None)


def test_install_covers_every_denoiser_module(stub_torch):
    pipe = types.SimpleNamespace(
        transformer = _FakeDiT(),
        transformer_2 = _FakeDiT(),
        unconditional_transformer = _FakeDiT(),
    )
    handles = cg.install_cuda_graphs(pipe)
    assert len(handles) == 3
    assert all(h.enabled for h in handles)


def test_second_wrapper_reuses_the_shared_pool(stub_torch):
    first, second = _armed(), _armed()
    first(_t(), timestep = _t((1,)), return_dict = False)
    pool = cg._POOL_BOX[0]
    assert pool is not None
    second(_t((2, 4)), timestep = _t((1,)), return_dict = False)

    recorded = stub_torch._records["graphs"]
    assert recorded[0][1] is None  # first capture seeds the pool
    assert recorded[1][1] == pool  # every later capture is handed it

    cg.uninstall_all([first, second])
    assert cg._POOL_BOX[0] is None  # last wrapper freed -> the pool id is forgotten


def test_pool_survives_while_another_wrapper_still_holds_a_graph(stub_torch):
    first, second = _armed(), _armed()
    first(_t(), timestep = _t((1,)), return_dict = False)
    second(_t(), timestep = _t((1,)), return_dict = False)
    pool = cg._POOL_BOX[0]
    cg.uninstall_all([first])
    assert cg._POOL_BOX[0] == pool


def test_reset_all_forgets_the_pool_token_with_the_last_graph(stub_torch):
    """A reset that destroys the last graph in the shared pool must forget its token, or the next
    capture dies on the allocator's "use_count > 0 INTERNAL ASSERT FAILED"."""
    first, second = _armed(), _armed()
    first(_t(), timestep = _t((1,)), return_dict = False)
    second(_t((2, 4)), timestep = _t((1,)), return_dict = False)
    pool = cg._POOL_BOX[0]
    assert pool is not None

    # One wrapper still holds a graph: the pool is live and the token stays.
    cg.reset_all([first])
    assert cg._POOL_BOX[0] == pool

    # Now nothing does.
    cg.reset_all([second])
    assert cg._POOL_BOX[0] is None

    # And the next capture seeds a fresh pool instead of replaying the dead token.
    first(_t(), timestep = _t((1,)), return_dict = False)
    assert stub_torch._records["graphs"][-1][1] is None


def test_signature_is_the_original_forwards(stub_torch):
    """H3 filters kwargs by ``inspect.signature``, so a ``(*args, **kwargs)`` wrapper drops them."""
    module = _FakeDiT()
    handle = _armed(module)
    expected = inspect.signature(_FakeDiT.forward.__get__(module))
    assert handle.__signature__ == expected
    assert inspect.signature(module.forward) == expected
    assert list(inspect.signature(module.forward).parameters) == [
        "hidden_states",
        "timestep",
        "return_dict",
    ]
    assert handle.__wrapped__ == handle.orig
    assert handle.__name__ == "forward"


@pytest.mark.parametrize(
    "token, disabled",
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("On", True),
        (" true ", True),
        ("0", False),
        ("", False),
        ("no", False),
    ],
)
def test_env_kill_switch(monkeypatch, token, disabled):
    monkeypatch.setenv(cg.CUDA_GRAPH_DISABLE_ENV, token)
    assert cg.cuda_graph_disabled() is disabled


def test_kill_switch_unset_is_enabled(monkeypatch):
    monkeypatch.delenv(cg.CUDA_GRAPH_DISABLE_ENV, raising = False)
    assert cg.cuda_graph_disabled() is False


def _target(*, device = "cuda", backend = "cuda"):
    return types.SimpleNamespace(device = device, backend = backend, dtype = "bfloat16")


def _eligible(monkeypatch, **overrides):
    monkeypatch.delenv(cg.CUDA_GRAPH_DISABLE_ENV, raising = False)
    kwargs = {
        "family": types.SimpleNamespace(),
        "pipe": types.SimpleNamespace(transformer = _FakeDiT()),
        "offload_active": False,
        "cache_active": False,
        "speed_mode": "default",
    }
    target = overrides.pop("target", _target())
    kwargs.update(overrides)
    return cg.graph_eligible(target, **kwargs)


def test_graph_eligible_happy_path(stub_torch, monkeypatch):
    ok, reason = _eligible(monkeypatch)
    assert ok is True
    assert reason == "eligible"
    assert _eligible(monkeypatch, speed_mode = "max")[0] is True


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"target": _target(device = "mps", backend = "mps")}, "device is mps"),
        ({"target": _target(device = "cpu", backend = "cpu")}, "device is cpu"),
        ({"target": _target(backend = "rocm")}, "backend is rocm"),
        ({"offload_active": True}, "offload active"),
        ({"cache_active": True}, "step cache active"),
        ({"speed_mode": "eager"}, "speed tier eager"),
        ({"speed_mode": "off"}, "speed tier off"),
        ({"speed_mode": None}, "speed tier off"),
        (
            {"pipe": types.SimpleNamespace(unet = UNet2DConditionModel(), transformer = _FakeDiT())},
            "denoiser is a U-Net",
        ),
        ({"pipe": types.SimpleNamespace()}, "no denoiser transformer"),
        (
            {"family": types.SimpleNamespace(supports_cuda_graph = False)},
            "family opts out",
        ),
        ({"family_default": False}, "family opts out"),
    ],
)
def test_graph_eligible_refusals(stub_torch, monkeypatch, overrides, reason):
    ok, got = _eligible(monkeypatch, **overrides)
    assert ok is False
    assert got == reason


def test_graph_eligible_refuses_under_the_kill_switch(stub_torch, monkeypatch):
    monkeypatch.setenv(cg.CUDA_GRAPH_DISABLE_ENV, "1")
    ok, reason = cg.graph_eligible(
        _target(),
        family = types.SimpleNamespace(),
        pipe = types.SimpleNamespace(transformer = _FakeDiT()),
        offload_active = False,
        cache_active = False,
        speed_mode = "default",
    )
    assert ok is False
    assert reason == f"disabled by {cg.CUDA_GRAPH_DISABLE_ENV}"


def test_graph_eligible_needs_torch_cuda(stub_torch, monkeypatch):
    stub_torch.cuda.is_available = lambda: False
    assert _eligible(monkeypatch)[1] == "torch.cuda unavailable"

    del stub_torch.cuda
    assert _eligible(monkeypatch)[1] == "torch.cuda unavailable"


def test_graph_eligible_family_opt_in_on_the_video_backend(stub_torch, monkeypatch):
    opted_in = types.SimpleNamespace(supports_cuda_graph = True)
    assert _eligible(monkeypatch, family = opted_in, family_default = False)[0] is True
    bare = types.SimpleNamespace()
    assert _eligible(monkeypatch, family = bare, family_default = False)[0] is False


def test_stats_and_describe_are_json_safe(stub_torch):
    module = _FakeDiT()
    handle = _armed(module)
    handle(_t(), timestep = _t((1,)), return_dict = False)
    handle(_t((2, 4)), timestep = _t((1,)), return_dict = False)
    handle(_t(), timestep = _t((1,)), return_dict = True)

    aggregate = cg.stats([handle])
    assert aggregate == {
        "graphs": 2,
        "captures": 2,
        "replays": 2,
        "eager_calls": 1,
        "fallbacks": 0,
        "cap_skips": 0,
        "poisoned": False,
        "capture_error": None,
    }
    assert json.loads(json.dumps(aggregate)) == aggregate

    described = handle.describe()
    assert described["module"] == "_FakeDiT"
    assert described["graphs"] == 2
    assert described["enabled"] is True
    assert json.loads(json.dumps(described)) == described


def test_stats_reports_the_capture_error_without_the_traceback(stub_torch):
    handle = _armed()
    handle(_t(device_type = "cpu"), timestep = _t((1,)), return_dict = False)
    aggregate = cg.stats([handle])

    assert aggregate["poisoned"] is True
    assert aggregate["capture_error"]["type"] == "RuntimeError"
    assert "traceback" not in aggregate["capture_error"]
    assert json.loads(json.dumps(aggregate)) == aggregate
    assert cg.stats(())["graphs"] == 0


def test_real_cuda_capture_replays_bit_identically():
    torch = pytest.importorskip("torch")
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    import torch.nn as nn

    class TinyDiT(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.proj_in = nn.Linear(dim, dim)
            self.act = nn.SiLU()
            self.proj_out = nn.Linear(dim, dim)

        def forward(
            self,
            hidden_states,
            timestep,
            return_dict = True,
        ):
            out = self.proj_out(self.act(self.proj_in(hidden_states))) + timestep
            if return_dict:
                return types.SimpleNamespace(sample = out)
            return (out,)

    torch.manual_seed(0)
    dim = 64
    module = TinyDiT(dim).to("cuda", torch.bfloat16).eval()
    handle = cg.GraphedForward(module).enable()
    try:
        for _ in range(3):
            hidden = torch.randn(2, 8, dim, device = "cuda", dtype = torch.bfloat16)
            timestep = torch.randn(1, 1, device = "cuda", dtype = torch.bfloat16)
            with torch.inference_mode():
                want = handle.orig(hidden_states = hidden, timestep = timestep, return_dict = False)[0]
                got = module(hidden_states = hidden, timestep = timestep, return_dict = False)[0]
            assert torch.equal(got, want)

        assert handle.stats["captures"] == 1
        assert handle.stats["replays"] == 3
        assert len(handle.cache) == 1

        # A second shape records a second graph rather than replaying the first.
        hidden = torch.randn(2, 16, dim, device = "cuda", dtype = torch.bfloat16)
        timestep = torch.randn(1, 1, device = "cuda", dtype = torch.bfloat16)
        with torch.inference_mode():
            want = handle.orig(hidden_states = hidden, timestep = timestep, return_dict = False)[0]
            got = module(hidden_states = hidden, timestep = timestep, return_dict = False)[0]
        assert torch.equal(got, want)
        assert handle.stats["captures"] == 2
        assert cg.stats([handle])["graphs"] == 2

        # return_dict = True stays eager and still returns the dataclass-shaped output.
        with torch.inference_mode():
            assert hasattr(module(hidden_states = hidden, timestep = timestep), "sample")
        assert handle.stats["eager_calls"] == 1

        cg.uninstall_all([handle])
        assert "forward" not in module.__dict__
        with torch.inference_mode():
            after = module(hidden_states = hidden, timestep = timestep, return_dict = False)[0]
        assert torch.equal(after, want)
    finally:
        cg.uninstall_all([handle])
