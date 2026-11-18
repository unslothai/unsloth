# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CUDA graph captures on a second card. torch.cuda.graph keeps ONE class-level default capture stream, created on the
first card that captured: a later stream-less capture on another card switches to that stream (and so to that card),
while the kernels launch on the card their tensors live on, so the graph records nothing. The fake torch below keeps
those rules (stream context switches the current card, a launch lands on its tensor's card's current stream, the
capture holds what landed on its own stream); the last test runs the real thing on two cards."""

from __future__ import annotations

import contextlib
import sys
import types

import pytest

from core.inference import diffusion_cuda_graph as cg


class _Dev:
    def __init__(
        self,
        kind,
        index = None,
    ):
        if isinstance(kind, _Dev):
            kind, index = kind.type, kind.index
        self.type, self.index = kind, index

    def __eq__(self, other):
        return isinstance(other, _Dev) and (self.type, self.index) == (other.type, other.index)

    def __hash__(self):
        return hash((self.type, self.index))


class _Stream:
    def __init__(
        self,
        device = None,
        state = None,
    ):
        state = state or _STATE[0]
        index = state.current if device is None else _index(device)
        self.device = _Dev("cuda", index)
        self.device_index = index

    def wait_stream(self, other):
        pass

    def synchronize(self):
        pass


def _index(device):
    if isinstance(device, _Dev):
        return device.index
    return int(device)


class _T:
    def __init__(self, device):
        self.device = device

    def to(self, device, *_a, **_k):
        return _T(_Dev(device) if not isinstance(device, _Dev) else device)


class _Graph:
    def __init__(self):
        self.kernels: list = []
        self.stream = None
        self.device = None
        self.replays = 0

    def replay(self):
        self.replays += 1


class _State:
    def __init__(self):
        self.current = 0
        self.streams: dict = {}  # card -> its current stream
        self.capturing = None  # (graph, stream) of the open capture
        self.graphs: list = []

    def stream_of(self, index):
        return self.streams.setdefault(index, _Stream(index, self))

    def launch(self, tensor):
        index = tensor.device.index
        stream = self.stream_of(index)
        if self.capturing is not None and self.capturing[1] is stream:
            self.capturing[0].kernels.append(index)


_STATE: list = [None]


def _fake_torch():
    state = _State()
    _STATE[0] = state
    torch = types.ModuleType("torch")
    torch.int8, torch.bfloat16 = "int8", "bfloat16"
    torch.device = _Dev

    class _Gen:
        def __init__(self, device = None):
            pass

        def manual_seed(self, _seed):
            return self

    torch.Generator = _Gen
    torch.randint = lambda *a, **k: _T(_Dev("cpu"))
    torch.randn = lambda *a, **k: _T(_Dev("cpu"))
    torch.full = lambda *a, device = None, **k: _T(_Dev(device))

    @contextlib.contextmanager
    def device(dev):
        prev = state.current
        state.current = _index(dev)
        try:
            yield
        finally:
            state.current = prev

    def set_stream(stream):
        # like torch._C._cuda_setStream: the stream's card becomes current
        state.current = stream.device_index
        state.streams[stream.device_index] = stream

    @contextlib.contextmanager
    def stream(s):
        prev_cur = state.current
        prev = state.stream_of(s.device_index)
        src = state.stream_of(prev_cur)
        set_stream(s)
        try:
            yield
        finally:
            state.streams[s.device_index] = prev
            set_stream(src)

    class graph:  # noqa: N801 - torch's spelling
        default_capture_stream = None

        def __init__(
            self,
            cuda_graph,
            pool = None,
            stream = None,
            capture_error_mode = "global",
        ):
            if stream is None and graph.default_capture_stream is None:
                graph.default_capture_stream = _Stream(None, state)
            self.cuda_graph = cuda_graph
            self.stream_ctx = torch.cuda.stream(
                stream if stream is not None else graph.default_capture_stream
            )

        def __enter__(self):
            self.stream_ctx.__enter__()
            s = state.stream_of(state.current)
            self.cuda_graph.stream, self.cuda_graph.device = s, state.current
            state.capturing = (self.cuda_graph, s)

        def __exit__(self, *exc):
            state.capturing = None
            state.graphs.append(self.cuda_graph)
            self.stream_ctx.__exit__(*exc)
            return False

    class _Event:
        def __init__(self, enable_timing = False):
            pass

        def record(self, *_a):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, _other):
            return 1.0

    torch.cuda = types.SimpleNamespace(
        device = device,
        Stream = _Stream,
        stream = stream,
        set_stream = set_stream,
        current_stream = lambda dev = None: state.stream_of(
            state.current if dev is None else _index(dev)
        ),
        current_device = lambda: state.current,
        CUDAGraph = _Graph,
        graph = graph,
        Event = _Event,
        synchronize = lambda *a, **k: None,
        empty_cache = lambda: None,
        is_available = lambda: True,
    )
    torch._state = state
    return torch


@pytest.fixture
def fake_torch(monkeypatch):
    torch = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(cg, "_CAPTURE_STREAMS", {})
    return torch


def _assert_each_card_recorded(state, cards):
    assert state.graphs, "nothing was captured"
    for g in state.graphs:
        assert g.kernels, (
            f"a graph captured for cuda:{g.device} recorded no kernel (the second card's capture ran on the "
            f"first card's default capture stream)"
        )
        assert set(g.kernels) == {g.device}, "a capture recorded another card's kernels"
    assert {g.device for g in state.graphs} == set(cards)


def test_int8_gemm_latency_probe_records_on_each_card(fake_torch):
    from core.inference import diffusion_int8_gemm as g8

    state = fake_torch._state
    fns = {
        "fused": lambda a, w, xs, ws: state.launch(a),
        "stock": lambda a, w, xs, ws: state.launch(a),
    }
    first = g8._graph_latency(0, fns)
    second = g8._graph_latency(1, fns)
    assert set(first) == set(second) == {"fused", "stock"}
    _assert_each_card_recorded(state, (0, 1))


def test_convrot_latency_probe_records_on_each_card(fake_torch, monkeypatch):
    from core.inference import diffusion_convrot as cv
    from core.inference import diffusion_convrot_quant as cq

    state = fake_torch._state
    monkeypatch.setattr(
        cv, "build_convrot_hadamard", lambda n, device = None, dtype = None: _T(_Dev(device))
    )
    monkeypatch.setattr(cv, "rotate_convrot_activation", lambda x, h, n: state.launch(x) or x)
    monkeypatch.setattr(cq, "_h16_launch", lambda x, variant, tf32x3 = False: state.launch(x) or x)
    cq._unlisted_latency(0, False)
    cq._unlisted_latency(1, False)
    _assert_each_card_recorded(state, (0, 1))


def test_card_graph_makes_the_card_current_and_reuses_one_stream(fake_torch):
    state = fake_torch._state
    seen = []
    for index in (1, 0, 1):
        graph = fake_torch.cuda.CUDAGraph()
        with cg.card_graph(graph, index):
            seen.append((state.current, graph.stream))
    assert [c for c, _ in seen] == [1, 0, 1]
    assert seen[0][1] is seen[2][1], "one capture stream per card"
    assert seen[0][1].device_index == 1 and seen[1][1].device_index == 0
    assert state.current == 0, "the caller's card is restored"


def _two_cards() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() >= 2
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _two_cards(), reason = "needs two CUDA cards")
def test_real_second_card_capture_is_not_empty():
    import torch
    def body(x):
        return (x * 2 + 1).sum()

    for index in (0, 1):
        dev = torch.device("cuda", index)
        x = torch.ones(1024, device = dev)
        with torch.cuda.device(dev):
            body(x)
            torch.cuda.synchronize(dev)
            graph = torch.cuda.CUDAGraph()
            with cg._capturing(), cg.card_graph(graph, index):
                out = body(x)
            x.fill_(3.0)
            graph.replay()
            torch.cuda.synchronize(dev)
        assert out.device == dev
        assert (
            float(out) == 1024 * 7.0
        ), f"cuda:{index} graph replay did not run the recorded kernels"
