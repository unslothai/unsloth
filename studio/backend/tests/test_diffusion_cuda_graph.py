# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Denoiser CUDA-graph tests run on a fake torch; only the last test uses real torch and needs CUDA."""

from __future__ import annotations

import contextlib
import inspect
import json
import sys
import threading
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
    def __init__(self, device = None):
        self.device = device
        self.waited: list = []

    def wait_stream(self, other):
        self.waited.append(other)


def _build_stub_torch():
    """Just enough torch for the capture path, plus a record of what the layer did to it."""
    torch = types.ModuleType("torch")
    records = {
        "graphs": [],
        "streams": [],
        "synchronize": 0,
        "empty_cache": 0,
        "inference_mode": [],
        "graph_error": None,
        "capture_streams": [],
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
    def _graph(
        graph,
        pool = None,
        stream = None,
        capture_error_mode = "global",
    ):
        records["graphs"].append((graph, pool))
        records["capture_mode"] = capture_error_mode
        records["capture_streams"].append(stream)
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
    cg._CAPTURE_STREAMS.clear()
    yield
    cg._POOL_BOX[0] = None
    cg._LIVE_WRAPPERS.clear()
    cg._CAPTURE_STREAMS.clear()


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
    assert base == cg.graph_key(_t((2, 4)))
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
    handle(_t(), timestep = _t((1,)))
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
    assert entry.static[0].copied_from == [first, first, second]
    assert entry.static[0].value == 2.0
    assert entry.graph.replays == 2
    assert module.calls == cg.WARMUP_ITERS + 1
    assert stub_torch._records["synchronize"] == 1
    assert stub_torch._records["inference_mode"] == [False]


def test_each_card_records_on_its_own_capture_stream(stub_torch):
    """torch's default capture stream lives on the first card that captured; each card needs its own."""
    first = _armed()
    first(_t(device_index = 0), timestep = _t((1,), device_index = 0), return_dict = False)
    second = _armed()
    second(_t(device_index = 1), timestep = _t((1,), device_index = 1), return_dict = False)
    third = _armed()
    third(_t(device_index = 1), timestep = _t((1,), device_index = 1), return_dict = False)
    streams = stub_torch._records["capture_streams"]
    assert [s.device for s in streams] == [0, 1, 1]
    assert (
        streams[1] is streams[2]
    ), "one stream per card: pool blocks are reused only from the recording stream"


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
    assert module.calls == 1
    assert out[0].value == ("out", 1)
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
    assert module.calls == cg.WARMUP_ITERS + 1
    assert out[0].value == ("out", module.calls)
    assert len(logged) == 1
    assert "%s" in logged[0][0] and "capture failed" in logged[0][0]
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
    """Once poisoned, ``__call__`` never reads the cache again, so its entries only pin memory."""
    module = _FakeDiT()
    handle = _armed(module)
    handle(_t((1, 4)), timestep = _t((1,)), return_dict = False)
    assert len(handle.cache) == 1
    assert cg._POOL_BOX[0] is not None

    stub_torch._records["graph_error"] = RuntimeError("CUDA out of memory during capture")
    out = handle(_t((2, 4)), timestep = _t((1,)), return_dict = False)

    assert handle.poisoned is True
    assert out[0].value[0] == "out"
    assert handle.cache == {}, "graphs that can never replay again are still pinning memory"
    assert cg._POOL_BOX[0] is None, "and the pool token outlived the last graph in it"
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
    assert len(logged) == 1

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
    assert len(handle.cache) == 1

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
    assert handle.stats["captures"] == 2


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
    cg.uninstall_all(handles)
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
    assert recorded[0][1] is None
    assert recorded[1][1] == pool

    cg.uninstall_all([first, second])
    assert cg._POOL_BOX[0] is None


def test_pool_survives_while_another_wrapper_still_holds_a_graph(stub_torch):
    first, second = _armed(), _armed()
    first(_t(), timestep = _t((1,)), return_dict = False)
    second(_t(), timestep = _t((1,)), return_dict = False)
    pool = cg._POOL_BOX[0]
    cg.uninstall_all([first])
    assert cg._POOL_BOX[0] == pool


def test_a_graph_in_an_earlier_pool_does_not_keep_the_current_token(stub_torch):
    """Liveness is per graph: a wrapper still holding a graph in a pool dropped earlier (a placement move re-recorded
    another wrapper into a fresh pool) must not keep the current token alive after its own last graph is gone."""
    first, second = _armed(), _armed()
    first(_t(), timestep = _t((1,)), return_dict = False)
    cg._POOL_BOX[0] = None
    second(_t(), timestep = _t((1,)), return_dict = False)
    current = cg._POOL_BOX[0]
    assert current is not None and first.cache
    cg.reset_all([second])
    assert cg._POOL_BOX[0] is None


def test_reset_all_forgets_the_pool_token_with_the_last_graph(stub_torch):
    """A reset that destroys the last graph in the shared pool must forget its token, or the next
    capture dies on the allocator's "use_count > 0 INTERNAL ASSERT FAILED"."""
    first, second = _armed(), _armed()
    first(_t(), timestep = _t((1,)), return_dict = False)
    second(_t((2, 4)), timestep = _t((1,)), return_dict = False)
    pool = cg._POOL_BOX[0]
    assert pool is not None

    cg.reset_all([first])
    assert cg._POOL_BOX[0] == pool

    cg.reset_all([second])
    assert cg._POOL_BOX[0] is None

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


def test_graph_eligible_accepts_a_unet_denoiser(stub_torch, monkeypatch):
    """A whole-compiled SDXL U-Net is captured at ``_compiled_call_impl`` (GraphedCompiledCall)."""
    assert (
        _eligible(monkeypatch, pipe = types.SimpleNamespace(unet = UNet2DConditionModel()))[0] is True
    )
    other = type("SomeOtherUNet", (), {})()
    assert (
        _eligible(monkeypatch, pipe = types.SimpleNamespace(unet = other))[1]
        == "no denoiser transformer"
    )


class _CompiledUNet:
    """``Module.compile``d stand-in: ``module(...)`` is served by ``_compiled_call_impl``."""

    def __init__(self):
        self.calls = 0

        def compiled(
            sample,
            timestep = None,
            return_dict = True,
        ):
            self.calls += 1
            return (_FakeTensor((1, 4), value = ("out", self.calls), tag = "out"),)

        self._compiled_call_impl = compiled

    def forward(self, *args, **kwargs):  # pragma: no cover - a compiled module never reads the slot
        raise AssertionError("the forward slot must not be used")


_CompiledUNet.__name__ = "UNet2DConditionModel"  # the class NAME is what _denoiser_unet gates on


def test_compiled_unet_is_graphed_at_the_compiled_call(stub_torch):
    unet = _CompiledUNet()
    compiled = unet._compiled_call_impl
    pipe = types.SimpleNamespace(unet = unet)
    handles = cg.install_cuda_graphs(pipe)
    assert len(handles) == 1 and isinstance(handles[0], cg.GraphedCompiledCall)
    assert unet._compiled_call_impl is handles[0]
    assert "forward" not in unet.__dict__

    for _ in range(2):
        unet._compiled_call_impl(_t(), timestep = _t((1,)), return_dict = False)
    assert handles[0].stats["captures"] == 1 and handles[0].stats["replays"] == 2
    assert unet.calls == cg.WARMUP_ITERS + 1

    unet._compiled_call_impl(_t(), timestep = _t((1,)))
    assert unet.calls == cg.WARMUP_ITERS + 2

    cg.uninstall_all(handles)
    assert unet._compiled_call_impl is compiled


def test_uncompiled_unet_is_graphed_at_the_forward_slot(stub_torch):
    class UNet2DConditionModel(_FakeDiT):  # noqa: N801
        pass

    unet = UNet2DConditionModel()
    handles = cg.install_cuda_graphs(types.SimpleNamespace(unet = unet))
    assert len(handles) == 1 and type(handles[0]) is cg.GraphedForward
    assert unet.__dict__["forward"] is handles[0]
    cg.uninstall_all(handles)


def test_graphed_compiled_call_needs_a_compiled_module(stub_torch):
    with pytest.raises(RuntimeError):
        cg.GraphedCompiledCall(_FakeDiT())


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
    offload_only = types.SimpleNamespace(offload_cuda_graph = True)
    assert _eligible(monkeypatch, family = offload_only, family_default = False)[0] is False
    assert (
        _eligible(monkeypatch, family = offload_only, family_default = False, offloaded = True)[0] is True
    )


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
        "invalidations": 0,
        "pool_bytes": aggregate["pool_bytes"],
        "planned_eager": 0,
        "shape_warmups": 0,
        "evictions": 0,
        "speed_eager": 0,
        "refused_float": 0,
        "refused_host_tensor": 0,
        "refused_object": 0,
        "placements": ["resident"],
        "poisoned": False,
        "capture_error": None,
        "eager_ms": None,
        "replay_ms": None,
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

        hidden = torch.randn(2, 16, dim, device = "cuda", dtype = torch.bfloat16)
        timestep = torch.randn(1, 1, device = "cuda", dtype = torch.bfloat16)
        with torch.inference_mode():
            want = handle.orig(hidden_states = hidden, timestep = timestep, return_dict = False)[0]
            got = module(hidden_states = hidden, timestep = timestep, return_dict = False)[0]
        assert torch.equal(got, want)
        assert handle.stats["captures"] == 2
        assert cg.stats([handle])["graphs"] == 2

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


class _FakeNVFP4Linear:
    """Named exactly as the real class, since ``is_nvfp4_flashinfer_linear`` gates on the NAME."""

    def __init__(
        self,
        *,
        baked = True,
        in_features = 4,
        out_features = 8,
    ):
        self.a_gsf = 1.0
        self.in_features = in_features
        self.out_features = out_features
        if baked is not None:
            self.activation_scales_baked = bool(baked)


_FakeNVFP4Linear.__name__ = "NVFP4FlashInferLinear"


class _NVFP4DiT(_FakeDiT):
    """A denoiser whose module tree holds the given NVFP4 linears."""

    def __init__(self, layers):
        super().__init__()
        self._layers = list(layers)

    def named_modules(self):
        yield "", self
        for name, layer in self._layers:
            yield name, layer


@pytest.fixture
def record_prewarm(monkeypatch):
    from core.inference import diffusion_nvfp4_linear as nl

    calls: list = []
    monkeypatch.setattr(
        nl,
        "nvfp4_prewarm",
        lambda transformer, shapes, **kwargs: calls.append((transformer, tuple(shapes))) or 0,
    )
    return calls


def test_capture_prewarms_the_nvfp4_layers_before_the_warmup(stub_torch, record_prewarm):
    module = _NVFP4DiT([("blocks.0.attention.to_q", _FakeNVFP4Linear())])
    handle = _armed(module)
    handle(_t((2, 8, 4)), timestep = _t((1,)), return_dict = False)

    assert len(record_prewarm) == 1
    tuned_module, shapes = record_prewarm[0]
    assert tuned_module is module
    assert 1 in shapes and 16 in shapes
    assert handle.stats["captures"] == 1
    assert handle.poisoned is False


def test_capture_does_not_prewarm_a_model_without_nvfp4_layers(stub_torch, record_prewarm):
    handle = _armed()
    handle(_t(), timestep = _t((1,)), return_dict = False)
    assert record_prewarm == []
    assert handle.stats["captures"] == 1


def test_unbaked_activation_scales_poison_the_capture(stub_torch, record_prewarm):
    module = _NVFP4DiT(
        [
            ("blocks.0.attention.to_q", _FakeNVFP4Linear()),
            ("blocks.1.attention.to_q", _FakeNVFP4Linear(baked = False)),
        ]
    )
    handle = _armed(module)
    out = handle(_t(), timestep = _t((1,)), return_dict = False)

    assert handle.poisoned is True
    assert handle.stats["captures"] == 0
    assert "unbaked activation scales" in handle.capture_error["msg"]
    assert "blocks.1.attention.to_q" in handle.capture_error["msg"]
    assert record_prewarm == []
    assert module.calls == 1
    assert out[0].value == ("out", 1)


def test_a_layer_that_cannot_answer_counts_as_unbaked(stub_torch, record_prewarm):
    """Fail closed: a layer from some other build with no flag is not assumed to be baked."""
    module = _NVFP4DiT([("blocks.0.attention.to_q", _FakeNVFP4Linear(baked = None))])
    handle = _armed(module)
    handle(_t(), timestep = _t((1,)), return_dict = False)
    assert handle.poisoned is True
    assert record_prewarm == []


def test_prewarm_token_counts_are_bounded_and_sorted(stub_torch):
    counts = cg._prewarm_token_counts([_t((2, 8, 4)), _t((1,)), _t((4, 4))])
    assert counts[0] == 1
    assert counts == tuple(sorted(set(counts)))
    assert set(counts) >= {1, 4, 16}
    assert len(cg._prewarm_token_counts([_t((n, 4)) for n in range(2, 40)])) <= 8


_RESOLVED_ON = {
    "cuda_graph": {
        "value": "on",
        "requested": None,
        "source": "auto",
        "status": "applied",
        "reason": "denoiser step captured per input shape, replayed bit-identically",
    },
    "speed_mode": {"value": "max"},
}


def test_live_status_is_unchanged_before_the_first_call(stub_torch):
    handle = _armed()
    resolved, optims = cg.live_status(_RESOLVED_ON, ("compiled", "cuda_graph"), (handle,))
    assert resolved is _RESOLVED_ON
    assert optims == ["compiled", "cuda_graph"]
    assert cg.never_engaged((handle,)) is None
    assert cg.never_engaged(()) is None


def test_live_status_turns_off_when_every_call_was_refused(stub_torch):
    class KVCache:
        pass

    handle = _armed()
    for _ in range(3):
        handle(_t(), timestep = KVCache(), return_dict = False)
    assert handle.stats["refused_object"] == 3

    resolved, optims = cg.live_status(_RESOLVED_ON, ("compiled", "cuda_graph"), (handle,))
    assert resolved["cuda_graph"]["value"] == "off"
    assert resolved["cuda_graph"]["reason"] == (
        "armed, but all 3 denoiser call(s) so far ran eager (3 with a non-tensor argument)"
    )
    assert resolved["cuda_graph"]["status"] == "applied"
    assert resolved["speed_mode"] is _RESOLVED_ON["speed_mode"]
    assert optims == ["compiled"]
    assert _RESOLVED_ON["cuda_graph"]["value"] == "on"
    assert json.loads(json.dumps(resolved)) == resolved


def test_live_status_names_float_refusals_and_a_poisoned_capture(stub_torch):
    handle = _armed()
    handle(_t(), timestep = 0.5, return_dict = False)
    assert "1 with a float argument" in cg.never_engaged((handle,))

    poisoned = _armed()
    poisoned(_t(device_type = "cpu"), timestep = _t((1,)), return_dict = False)
    assert cg.never_engaged((poisoned,)) == (
        "capture failed (RuntimeError); every denoiser step runs eager"
    )


def test_live_status_turns_off_when_a_graph_that_replayed_is_poisoned(stub_torch):
    handle = _armed()
    handle(_t(), timestep = _t((1,)), return_dict = False)
    handle(_t(), timestep = _t((1,)), return_dict = False)
    assert handle.stats["replays"] == 2 and cg.never_engaged((handle,)) is None
    handle(_t(device_type = "cpu"), timestep = _t((1,)), return_dict = False)
    assert handle.poisoned
    resolved, optims = cg.live_status(_RESOLVED_ON, ("compiled", "cuda_graph"), (handle,))
    assert resolved["cuda_graph"]["value"] == "off"
    assert resolved["cuda_graph"]["reason"] == (
        "capture failed (RuntimeError); every denoiser step runs eager"
    )
    assert optims == ["compiled"]
    other = _armed()
    other(_t(), timestep = _t((1,)), return_dict = False)
    assert cg.never_engaged((handle, other)) is None


def test_live_status_keeps_on_once_a_graph_replayed(stub_torch):
    handle = _armed()
    handle(_t(), timestep = _t((1,)), return_dict = False)
    handle(_t(), timestep = _t((1,)), return_dict = False)
    handle(_t(), timestep = 0.25, return_dict = False)
    assert handle.stats["replays"] == 2
    resolved, optims = cg.live_status(_RESOLVED_ON, ("cuda_graph",), (handle,))
    assert resolved is _RESOLVED_ON
    assert optims == ["cuda_graph"]


def test_live_status_tolerates_a_record_without_the_control(stub_torch):
    handle = _armed()
    handle(_t(), timestep = 0.5, return_dict = False)
    assert cg.live_status(None, ("cuda_graph",), (handle,)) == (None, [])
    assert cg.live_status({}, (), (handle,)) == ({}, [])


def test_real_cuda_compiled_module_capture_matches_the_compiled_call():
    torch = pytest.importorskip("torch")
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    try:
        import triton  # noqa: F401
    except Exception:
        pytest.skip("needs triton for inductor")

    import torch.nn as nn

    class UNet2DConditionModel(nn.Module):  # noqa: N801
        def __init__(self, dim):
            super().__init__()
            self.a = nn.Linear(dim, dim)
            self.b = nn.Linear(dim, dim)

        def forward(
            self,
            sample,
            timestep,
            return_dict = True,
        ):
            out = self.b(torch.nn.functional.silu(self.a(sample))) * timestep
            return types.SimpleNamespace(sample = out) if return_dict else (out,)

    torch.manual_seed(0)
    unet = UNet2DConditionModel(64).to("cuda", torch.bfloat16).eval()
    unet.compile(fullgraph = True, dynamic = False)
    compiled = unet._compiled_call_impl
    handles = cg.install_cuda_graphs(types.SimpleNamespace(unet = unet))
    try:
        assert isinstance(handles[0], cg.GraphedCompiledCall)
        for _ in range(3):
            x = torch.randn(2, 8, 64, device = "cuda", dtype = torch.bfloat16)
            t = torch.randn((), device = "cuda", dtype = torch.bfloat16)
            with torch.inference_mode():
                want = compiled(x, t, return_dict = False)[0]
                got = unet(x, t, return_dict = False)[0]
            assert torch.equal(got, want)
        assert handles[0].stats["captures"] == 1 and handles[0].stats["replays"] == 3
    finally:
        cg.uninstall_all(handles)
    assert unet._compiled_call_impl is compiled


@pytest.mark.parametrize("backend", ["sage", "sage_hub"])
def test_graph_eligible_declines_a_sage_denoiser(stub_torch, monkeypatch, backend):
    """SageAttention under a replayed graph renders noise (FLUX.1-schnell, A100), so a Sage load stays ungraphed; the
    kernels-hub build (sage_hub) is the same kernel."""
    dit = _FakeDiT()
    dit._unsloth_attention_backend = backend
    ok, reason = _eligible(monkeypatch, pipe = types.SimpleNamespace(transformer = dit))
    assert (ok, reason) == (False, "SageAttention is not CUDA-graph safe")


@pytest.mark.parametrize("backend", [None, "_native_cudnn", "flash", "native"])
def test_graph_eligible_keeps_other_backends(stub_torch, monkeypatch, backend):
    dit = _FakeDiT()
    dit._unsloth_attention_backend = backend
    assert _eligible(monkeypatch, pipe = types.SimpleNamespace(transformer = dit)) == (True, "eligible")


def _named_dit(name):
    return type(name, (_FakeDiT,), {})()


def _policy():
    from core.inference.diffusion_dynamic_text import PROMPT_LENGTH_POLICY
    return PROMPT_LENGTH_POLICY


def _step(
    handle,
    text_len,
    value = 0.0,
):
    """One denoiser call whose text stream is ``text_len`` tokens (the graph key follows it)."""
    return handle(
        _t((1, 64), value = value), encoder_hidden_states = _t((1, text_len)), return_dict = False
    )


def _render(
    handle,
    text_len,
    steps = 3,
    negative_len = 7,
):
    """A true-CFG render: per step the prompt's call, then the negative prompt's."""
    for i in range(steps):
        _step(handle, text_len, value = float(i))
        _step(handle, negative_len, value = float(i))


class _LenDiT:
    """``_FakeDiT`` that also takes the text stream."""

    def __init__(self):
        self.calls = 0

    def forward(
        self,
        hidden_states,
        encoder_hidden_states = None,
        return_dict = True,
    ):
        self.calls += 1
        return (_FakeTensor((1, 64), value = ("out", hidden_states.value), tag = "out"),)


def _len_dit(name):
    return type(name, (_LenDiT,), {})()


@pytest.mark.parametrize("name", sorted(_policy()))
def test_the_wrapper_reads_the_prompt_length_policy(stub_torch, monkeypatch, name):
    monkeypatch.delenv(cg.EVERY_LENGTH_ENV, raising = False)
    handle = _armed(_len_dit(name))
    assert handle.length_varies is isinstance(_policy()[name], tuple)
    assert handle.length_sealed is False
    # whole-step graphs stay eligible for every family: only what they record changes
    ok, reason = _eligible(monkeypatch, pipe = types.SimpleNamespace(transformer = _named_dit(name)))
    assert (ok, reason) == (True, "eligible")


def test_a_new_prompt_length_after_the_first_render_records_nothing(stub_torch, monkeypatch):
    """Qwen-Image's text length follows the prompt: generation 2 with a new length recorded a graph on that render
    (seconds of capture). The first render's lengths replay; a new one runs the step ungraphed."""
    monkeypatch.delenv(cg.EVERY_LENGTH_ENV, raising = False)
    module = _len_dit("QwenImageTransformer2DModel")
    handle = _armed(module)
    _render(handle, 3)
    assert handle.stats["captures"] == 2 and handle.length_sealed is True
    calls, replays, graphs = module.calls, handle.stats["replays"], stub_torch._records["graphs"][:]

    _render(handle, 15)
    assert handle.stats["captures"] == 2
    assert stub_torch._records["graphs"] == graphs
    assert handle.stats["length_eager"] == 3
    assert module.calls == calls + 3
    assert handle.stats["replays"] == replays + 3
    assert len(handle.cache) == 2 and handle.stats["evictions"] == 0

    _render(handle, 3)
    assert handle.stats["captures"] == 2 and handle.stats["replays"] == replays + 9
    assert handle.stats["length_eager"] == 3


def _planned_len_handle(stub_torch, monkeypatch):
    import contextlib

    monkeypatch.delenv(cg.EVERY_LENGTH_ENV, raising = False)
    handle = _armed(_len_dit("QwenImage21Transformer2DModel"))
    orig = handle.orig
    handle.plan = lambda mod, args, kwargs: (orig, {"hidden_states": args[0], **kwargs})
    monkeypatch.setattr(_FakeTensor, "is_inference", lambda self: False, raising = False)
    monkeypatch.setattr(
        stub_torch, "inference_mode", lambda *a, **k: contextlib.nullcontext(), raising = False
    )
    monkeypatch.setattr(cg, "_graph_without_flush", stub_torch.cuda.graph)
    return handle


def test_a_planned_negative_of_another_length_records_on_the_first_render(stub_torch, monkeypatch):
    """The negative is still warming when the positive's replay seals the scope; it must still record."""
    handle = _planned_len_handle(stub_torch, monkeypatch)
    for _ in range(2):
        _render(handle, 3, steps = 4)
        cg.seal_lengths((handle,))
    assert handle.length_sealed is True
    assert handle.stats["captures"] == 2 and handle.stats["shape_warmups"] == 1
    assert handle.stats["length_eager"] == 0
    assert handle.stats["replays"] == 2 * 4 * 2 - 1
    _render(handle, 15, steps = 2, negative_len = 3)
    assert handle.stats["captures"] == 2 and handle.stats["length_eager"] == 2


def test_a_warmup_from_before_a_reset_does_not_reopen_the_sealed_scope(stub_torch, monkeypatch):
    handle = _planned_len_handle(stub_torch, monkeypatch)
    _step(handle, 3)
    _step(handle, 7)
    assert handle.stats["captures"] == 1 and handle.stats["shape_warmups"] == 1
    handle.reset()
    for i in range(3):
        _step(handle, 3, value = float(i))
    cg.seal_lengths((handle,))
    assert handle.length_sealed is True
    captures = handle.stats["captures"]
    for i in range(3):
        _step(handle, 7, value = float(i))
    assert handle.stats["captures"] == captures
    assert handle.stats["length_eager"] == 3


def test_a_warmup_from_an_earlier_render_does_not_reopen_the_sealed_scope(stub_torch, monkeypatch):
    handle = _planned_len_handle(stub_torch, monkeypatch)
    _step(handle, 3)
    _step(handle, 7)
    cg.seal_lengths((handle,))
    assert handle.length_sealed is True
    _render(handle, 3, steps = 2)
    assert handle.stats["captures"] == 1 and handle.stats["length_eager"] == 2


def test_the_ungraphed_step_returns_what_the_step_itself_returns(stub_torch, monkeypatch):
    monkeypatch.delenv(cg.EVERY_LENGTH_ENV, raising = False)
    handle = _armed(_len_dit("ZImageTransformer2DModel"))
    _render(handle, 32, negative_len = 32)
    out = _step(handle, 64, value = 5.0)
    assert out[0].value == ("out", 5.0) and out[0].clone_of is None


def test_a_fixed_length_family_records_as_before(stub_torch):
    handle = _armed(_len_dit("Flux2Transformer2DModel"))
    _render(handle, 512, negative_len = 512)
    _render(handle, 300, negative_len = 300)  # (cannot happen for FLUX.2: its text is padded to 512)
    # what it did before this policy: every new shape records (behaviour only, so it holds on either side)
    assert handle.stats["captures"] == 2 and handle.stats.get("length_eager", 0) == 0
    assert not getattr(handle, "length_sealed", False)


def test_reset_reopens_the_first_render(stub_torch, monkeypatch):
    monkeypatch.delenv(cg.EVERY_LENGTH_ENV, raising = False)
    handle = _armed(_len_dit("QwenImageTransformer2DModel"))
    _render(handle, 3)
    handle.reset()  # a LoRA changed the weights: nothing recorded is left to replay
    assert handle.length_sealed is False
    _render(handle, 15)
    assert handle.stats["captures"] == 4 and handle.stats["length_eager"] == 0
    assert handle.length_sealed is True


def test_the_every_length_switch_records_each_new_length(stub_torch, monkeypatch):
    monkeypatch.setenv(cg.EVERY_LENGTH_ENV, "1")
    handle = _armed(_len_dit("QwenImageTransformer2DModel"))
    _render(handle, 3)
    _render(handle, 15)
    assert handle.length_varies is False and handle.stats["captures"] == 3


def test_seal_lengths_closes_a_first_render_that_never_replayed(stub_torch, monkeypatch):
    monkeypatch.delenv(cg.EVERY_LENGTH_ENV, raising = False)
    varying = _armed(_len_dit("QwenImageTransformer2DModel"))
    fixed = _armed(_len_dit("Flux2Transformer2DModel"))
    idle = _armed(_len_dit("Flux3Transformer2DModel"))
    _step(varying, 3)
    _step(fixed, 512)
    assert varying.length_sealed is False
    cg.seal_lengths((varying, fixed, idle, object()))
    assert varying.length_sealed is True
    assert fixed.length_sealed is False
    assert idle.length_sealed is False
    _step(varying, 15)
    assert varying.stats["captures"] == 1 and varying.stats["length_eager"] == 1


def test_describe_reports_the_seal(stub_torch, monkeypatch):
    monkeypatch.delenv(cg.EVERY_LENGTH_ENV, raising = False)
    handle = _armed(_len_dit("QwenImageTransformer2DModel"))
    _render(handle, 3)
    _render(handle, 15)
    info = handle.describe()
    assert info["length_sealed"] is True and info["stats"]["length_eager"] == 3
    json.dumps(info)


def test_block_graphs_decline_a_variable_length_denoiser(stub_torch, monkeypatch):
    monkeypatch.delenv(cg.FORCE_FAMILIES_ENV, raising = False)
    pipe = types.SimpleNamespace(transformer = _named_dit("QwenImageTransformer2DModel"))
    assert cg.prompt_length_reason(pipe).startswith(cg.PROMPT_LENGTH_REASON)
    assert (
        cg.prompt_length_reason(
            types.SimpleNamespace(transformer = _named_dit("Flux2Transformer2DModel"))
        )
        is None
    )
    family = types.SimpleNamespace(name = "qwen-image")
    monkeypatch.setenv(cg.FORCE_FAMILIES_ENV, "qwen-image")
    assert cg.prompt_length_reason(pipe, family) is None


def _render_at(
    handle,
    width,
    text_len,
    steps = 2,
    negative_len = 7,
):
    """A true-CFG render at latent width ``width`` (the image scope) with a ``text_len``-token prompt."""
    for i in range(steps):
        handle(
            _t((1, width), value = float(i)),
            encoder_hidden_states = _t((1, text_len)),
            return_dict = False,
        )
        handle(
            _t((1, width), value = float(i)),
            encoder_hidden_states = _t((1, negative_len)),
            return_dict = False,
        )


def test_a_new_resolution_still_records_after_the_seal(stub_torch, monkeypatch):
    """The seal stops new TEXT lengths only: a later render at a new resolution (or batch) is a new image scope and
    records its first render, as before; a further new prompt length at that resolution then runs ungraphed."""
    monkeypatch.delenv(cg.EVERY_LENGTH_ENV, raising = False)
    handle = _armed(_len_dit("QwenImageTransformer2DModel"), max_graphs = 8)
    _render_at(handle, 64, 3)
    assert handle.stats["captures"] == 2 and handle.length_sealed
    _render_at(handle, 128, 15)
    assert handle.stats["captures"] == 4 and handle.stats["length_eager"] == 0
    replays = handle.stats["replays"]
    _render_at(handle, 128, 40)
    assert handle.stats["captures"] == 4 and handle.stats["length_eager"] == 2
    assert handle.stats["replays"] == replays + 2
    _render_at(handle, 64, 40)
    assert handle.stats["captures"] == 4 and handle.stats["length_eager"] == 4


def test_a_render_with_no_capture_after_reset_does_not_seal(stub_torch, monkeypatch):
    """seal_lengths counts captures since the last reset: a first render after reset that recorded nothing (bypassed
    here, or a planned step that only warmed up) must not seal an empty cache, or its key runs ungraphed forever."""
    monkeypatch.delenv(cg.EVERY_LENGTH_ENV, raising = False)
    handle = _armed(_len_dit("QwenImageTransformer2DModel"))
    _render(handle, 3)
    handle.reset()
    handle.set_bypass(True)
    _render(handle, 15)
    handle.set_bypass(False)
    cg.seal_lengths((handle,))
    assert handle.length_sealed is False
    _render(handle, 15)
    assert handle.stats["captures"] == 4 and handle.stats["length_eager"] == 0


def test_a_step_dependent_key_seals_only_when_the_render_ends(stub_torch, monkeypatch):
    """A key that also follows the step (NVFP4's protect branch, a pre-hook's branch flag) gets a new variant on a
    later step of the first render: a replay on step 2 must not seal the scope before that variant records."""
    monkeypatch.delenv(cg.EVERY_LENGTH_ENV, raising = False)
    module = _len_dit("QwenImageTransformer2DModel")
    branch = {"v": "a"}
    setattr(module, cg.GRAPH_KEY_EXTRA_ATTR, lambda: branch["v"])
    handle = _armed(module, max_graphs = 8)
    for i, flag in enumerate(("a", "a", "b", "b")):
        branch["v"] = flag
        _step(handle, 3, value = float(i))
    assert handle.stats["captures"] == 2 and handle.stats["length_eager"] == 0
    assert handle.length_sealed is False
    cg.seal_lengths((handle,))
    assert handle.length_sealed is True
    _step(handle, 15)
    assert handle.stats["captures"] == 2 and handle.stats["length_eager"] == 1


class _RefDiT(_LenDiT):
    def forward(
        self,
        hidden_states,
        encoder_hidden_states = None,
        image_cond_hidden_states = None,
        image_cond_ids = None,
        return_dict = True,
    ):
        return super().forward(hidden_states, encoder_hidden_states, return_dict)


def test_separate_reference_images_are_part_of_the_image_scope(stub_torch, monkeypatch):
    """FLUX.3 passes its reference images apart from the latents: a new reference layout at the same output size is a
    new scope that records its first render, not a sealed one that runs ungraphed."""
    monkeypatch.delenv(cg.EVERY_LENGTH_ENV, raising = False)
    module = type("Flux3Transformer2DModel", (_RefDiT,), {})()
    handle = _armed(module, max_graphs = 8)
    assert handle.length_varies is True

    def call(
        text_len,
        refs,
        value = 0.0,
    ):
        kw = {}
        if refs:
            kw = {"image_cond_hidden_states": _t((1, refs)), "image_cond_ids": _t((refs, 3))}
        return handle(
            _t((1, 64), value = value),
            encoder_hidden_states = _t((1, text_len)),
            return_dict = False,
            **kw,
        )

    for i in range(2):
        call(3, 0, float(i))  # text-only render: records, then replays and seals its scope
    assert handle.stats["captures"] == 1 and handle.length_sealed
    for i in range(2):
        call(3, 16, float(i))
    assert handle.stats["captures"] == 2 and handle.stats["length_eager"] == 0
    call(9, 16)
    assert handle.stats["captures"] == 2 and handle.stats["length_eager"] == 1


def test_the_every_length_switch_keeps_block_graphs(stub_torch, monkeypatch):
    monkeypatch.delenv(cg.FORCE_FAMILIES_ENV, raising = False)
    pipe = types.SimpleNamespace(transformer = _named_dit("QwenImageTransformer2DModel"))
    monkeypatch.delenv(cg.EVERY_LENGTH_ENV, raising = False)
    assert cg.prompt_length_reason(pipe) is not None
    monkeypatch.setenv(cg.EVERY_LENGTH_ENV, "1")
    assert cg.prompt_length_reason(pipe) is None


def test_real_cuda_new_length_eager_step_hands_blocks_the_recorded_tensor_kind():
    """A new-length eager step must not recompile compiled blocks by passing them inference tensors."""
    torch = pytest.importorskip("torch")
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    import torch.nn as nn
    from torch._dynamo.testing import CompileCounter

    torch._dynamo.reset()
    counter = CompileCounter()

    class Block(nn.Module):
        def forward(self, x, mod):
            return x * (1 + mod) + mod

    class LenDiT(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.proj = nn.Linear(dim, dim)
            self.block = torch.compile(Block(), backend = counter, dynamic = False)

        def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            mod,
            return_dict = True,
        ):
            x = self.proj(hidden_states) + encoder_hidden_states.mean()
            out = self.block(x, mod)
            return types.SimpleNamespace(sample = out) if return_dict else (out,)

    dim = 32
    torch.manual_seed(0)
    module = type("Flux3Transformer2DModel", (LenDiT,), {})(dim).to("cuda").eval()
    handle = cg.GraphedForward(module).enable()
    assert handle.length_varies
    hidden = torch.randn(1, 16, dim, device = "cuda")

    def render(text_len):
        outs = []
        with torch.inference_mode():
            txt = torch.randn(1, text_len, dim, device = "cuda")
            mod = torch.randn(1, 16, dim, device = "cuda")
            x = module.proj(hidden) + txt.mean()
            want = x * (1 + mod) + mod
            for _ in range(3):
                outs.append((module(hidden, txt, mod, return_dict = False)[0], want))
        cg.seal_lengths([handle])
        return outs

    try:
        render(5)
        assert handle.stats["captures"] == 1 and handle.stats["replays"] == 3
        frames = counter.frame_count
        for got, want in render(9):
            assert torch.allclose(got, want)
        assert handle.stats["length_eager"] == 3 and handle.stats["captures"] == 1
        # Before: the block recompiled for an inference ``mod``.
        assert counter.frame_count == frames
    finally:
        cg.uninstall_all([handle])


def test_new_length_eager_step_hands_blocks_normal_tensors_on_cpu(monkeypatch):
    """CPU twin of the CUDA test above, through the real ``__call__``."""
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "_dynamo"):
        pytest.skip("needs torch._dynamo")
    import torch.nn as nn
    from torch._dynamo.testing import CompileCounter

    monkeypatch.delenv(cg.EVERY_LENGTH_ENV, raising = False)
    torch._dynamo.reset()
    counter = CompileCounter()
    seen: list = []

    class Block(nn.Module):
        def forward(self, x, mod):
            return x * (1 + mod) + mod

    block = torch.compile(Block(), backend = counter, dynamic = False)

    class LenDiT(nn.Module):
        def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            mod,
            return_dict = True,
        ):
            seen.append(mod.is_inference())
            return (block(hidden_states + encoder_hidden_states.mean(), mod),)

    module = type("Flux3Transformer2DModel", (LenDiT,), {})()
    handle = _armed(module)
    assert handle.length_varies and handle.plan is None
    hidden = torch.randn(1, 16, 8)
    try:
        with torch.inference_mode():
            txt, mod = torch.randn(1, 9, 8), torch.randn(1, 16, 8)
            statics = handle._statics([hidden, txt, mod])
            handle.orig(*statics, return_dict = False)
            frames = counter.frame_count
            handle._sealed_scopes.add(cg._image_scope((hidden,), {}))
            got = module(hidden, txt, mod, return_dict = False)[0]
        assert handle.stats["length_eager"] == 1 and handle.stats["captures"] == 0
        # Before: the block recompiled for an inference ``mod``.
        assert counter.frame_count == frames
        assert seen == [False, False]
        assert torch.equal(got, (hidden + txt.mean()) * (1 + mod) + mod)
    finally:
        cg.uninstall_all([handle])


def test_eager_as_recorded_runs_the_original_call_when_the_copies_do_not_fit(monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    seen: list = []

    class LenDiT(nn.Module):
        def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            return_dict = True,
        ):
            seen.append(encoder_hidden_states.is_inference())
            return (hidden_states + encoder_hidden_states.mean(),)

    module = type("Flux3Transformer2DModel", (LenDiT,), {})()
    handle = _armed(module)

    def no_room(tensors):
        raise torch.cuda.OutOfMemoryError("no room")

    monkeypatch.setattr(handle, "_statics", no_room)
    try:
        with torch.inference_mode():
            hidden, txt = torch.randn(1, 4, 8), torch.randn(1, 3, 8)
            got = handle._eager_as_recorded((hidden, txt), {"return_dict": False})[0]
        assert seen == [True]
        assert torch.equal(got, hidden + txt.mean())
    finally:
        cg.uninstall_all([handle])


def test_a_refused_capture_runs_eager_and_records_on_a_later_call(stub_torch, monkeypatch):
    monkeypatch.setattr(cg, "OUTSIDE_WAIT_S", 0.05)
    retired = []
    real_retire = cg.retire_failed_capture
    monkeypatch.setattr(
        cg, "retire_failed_capture", lambda *a, **k: retired.append(a) or real_retire(*a, **k)
    )
    module = _FakeDiT()
    handle = _armed(module)
    cg.capture_hold()  # replica-side work that does not drain
    try:
        handle(_t(), timestep = _t((1,)), return_dict = False)
        assert handle.poisoned is False and handle.stats["captures"] == 0
        assert handle.stats["capture_refusals"] == 1 and handle.stats["eager_calls"] == 1
        assert retired == [] and not handle.cache and cg._CAPTURE_DEPTH == 0
        calls = module.calls
        # while the count stands a capture is refused up front, warm-ups included
        handle(_t(), timestep = _t((1,)), return_dict = False)
        assert handle.stats["capture_refusals"] == 2 and module.calls == calls + 1
    finally:
        cg.capture_release()
    calls = module.calls
    handle(_t(), timestep = _t((1,)), return_dict = False)
    assert handle.poisoned is False and handle.stats["captures"] == 1
    assert module.calls == calls + cg.WARMUP_ITERS + 1
    assert retired == []


def test_capture_warmups_wait_for_held_replica_work(stub_torch):
    seen = []

    class Dit(_FakeDiT):
        def forward(self, *args, **kwargs):
            seen.append(cg._OUTSIDE_JOBS)
            return super().forward(*args, **kwargs)

    handle = _armed(Dit())
    held, release = threading.Event(), threading.Event()

    def replica_job():
        with cg.outside_capture():
            held.set()
            release.wait(5)

    worker = threading.Thread(target = replica_job)
    worker.start()
    assert held.wait(5)
    timer = threading.Timer(0.3, release.set)
    timer.start()
    handle(_t(), timestep = _t((1,)), return_dict = False)
    worker.join(5)
    timer.join(5)
    assert handle.stats["captures"] == 1
    assert seen == [0] * (cg.WARMUP_ITERS + 1)
    assert cg._OUTSIDE_JOBS == 0 and cg._CAPTURE_DEPTH == 0


def test_engagement_reports_captures_replays_and_per_card_segments():
    """Status shows whether the graph layers are ENGAGED, not only that nothing recompiled."""
    import types as _types

    seg = _types.SimpleNamespace(stats = {"segment_captures": 2, "segment_replays": 30})
    whole = _types.SimpleNamespace(
        cache = {1: None},
        stats = {"captures": 1, "replays": 7, "eager_calls": 2, "invalidations": 0},
        placement = None,
        poisoned = False,
        capture_error = None,
    )
    blocks = _types.SimpleNamespace(
        cache = {},
        stats = {"captures": 3, "replays": 9},
        placement = None,
        poisoned = False,
        capture_error = None,
        segments = [seg],
    )
    out = cg.engagement((whole, blocks))
    assert out["graphs"] == 1 and out["captures"] == 4 and out["replays"] == 16
    assert out["eager_calls"] == 2 and out["segment_captures"] == 2 and out["segment_replays"] == 30
    assert set(out) >= {"dynamo_frames", "graph_breaks"}
    assert cg.engagement(())["captures"] == 0


def test_engagement_counts_the_per_block_graphs_a_whole_step_graph_fell_back_to():
    import types as _types

    seg = _types.SimpleNamespace(stats = {"segment_captures": 1, "segment_replays": 12})
    blocks = _types.SimpleNamespace(
        cache = {1: None, 2: None},
        stats = {"captures": 5, "replays": 40},
        placement = None,
        poisoned = False,
        capture_error = None,
        segments = [seg],
    )
    step = _types.SimpleNamespace(
        cache = {},
        stats = {"captures": 1, "replays": 2, "eager_calls": 3, "fallbacks": 1},
        placement = None,
        poisoned = True,
        capture_error = None,
        fallback_handle = blocks,
    )
    out = cg.engagement((step,))
    assert out["graphs"] == 2 and out["captures"] == 6 and out["replays"] == 42
    assert out["fallbacks"] == 1 and out["segment_captures"] == 1 and out["segment_replays"] == 12


def test_status_response_models_keep_graph_engagement():
    from models.inference import DiffusionStatusResponse, VideoStatusResponse
    counters = {"graphs": 1, "captures": 1, "replays": 7, "eager_calls": 0}
    for model in (DiffusionStatusResponse, VideoStatusResponse):
        dumped = model.model_validate({"graph_engagement": counters}).model_dump()
        assert dumped["graph_engagement"] == counters
        assert model.model_validate({}).model_dump()["graph_engagement"] is None
