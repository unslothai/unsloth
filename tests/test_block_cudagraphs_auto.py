# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Block CUDA graphs auto mode (UNSLOTH_BLOCK_CUDAGRAPHS unset): graphs turn on only after the first
forward passes all had one batch shape and a timed block launched slower than the GPU ran it.
"1" still forces them on, "0" (or any other value) keeps them off. Timing and the device checks are
stubbed, so this runs on CPU."""

import importlib.util
import os
import sys
import types

import pytest
import torch

_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth", "models"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "_ublock_compile_cg_auto", os.path.join(_DIR, "_block_compile.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bc = _load()


@pytest.fixture(autouse = True)
def _block_env(monkeypatch):
    for key in (
        "UNSLOTH_BLOCK_COMPILE",
        "UNSLOTH_COMPILE_DISABLE",
        "UNSLOTH_FORCE_FLOAT32",
        "UNSLOTH_BLOCK_CUDAGRAPHS",
    ):
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setattr(bc, "_CUDAGRAPH_STATE", bc._new_cudagraph_state())
    monkeypatch.setattr(bc, "_CUDAGRAPH_MODE", "auto")


@pytest.mark.parametrize(
    "value, mode",
    [(None, "auto"), ("", "auto"), ("auto", "auto"), ("1", "on"), ("0", "off"), ("true", "off")],
)
def test_mode_from_environment(monkeypatch, value, mode):
    """Unset is auto now; "1" forces graphs, every other value keeps the old meaning (off)."""
    try:
        if value is None:
            monkeypatch.delenv("UNSLOTH_BLOCK_CUDAGRAPHS", raising = False)
        else:
            monkeypatch.setenv("UNSLOTH_BLOCK_CUDAGRAPHS", value)
        mod = _load()
        assert mod._CUDAGRAPH_MODE == mode
        # Only a forced "1" records graphs from the first call; auto starts graphs-off.
        assert mod._COMPILE_OPTIONS["triton.cudagraphs"] is (mode == "on")
    finally:
        monkeypatch.delenv("UNSLOTH_BLOCK_CUDAGRAPHS", raising = False)
        _load()


def _harness(
    monkeypatch,
    host = 3e-3,
    gpu = 1e-3,
    free_gb = 40.0,
    static = None,
):
    calls = []
    compiles = []

    def fake_compile(fn, **kwargs):
        tag = "graphs" if kwargs.get("options", {}).get("triton.cudagraphs") else "plain"
        compiles.append(tag)

        def run(*args, **kw):
            calls.append(tag)
            return fn(*args, **kw)

        return run

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(bc, "_auto_static_reason", lambda layer, hs: static)
    timed = []

    def fake_time(call, device):
        timed.append(1)
        return call(), host, gpu

    monkeypatch.setattr(bc, "_auto_time", fake_time)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device = None: (int(free_gb * 2**30), 80 * 2**30)
    )

    class AutoDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    bc._wrap(AutoDecoderLayer, dict(bc._COMPILE_OPTIONS))
    layers = torch.nn.ModuleList([AutoDecoderLayer(), AutoDecoderLayer()]).train()
    return layers, calls, compiles, timed


def _step(
    layers,
    calls,
    L = 16,
    checkpoint = False,
    **kwargs,
):
    calls.clear()
    x = torch.randn(2, L, 8, requires_grad = True)
    h = x
    for layer in layers:
        if checkpoint:
            h = torch.utils.checkpoint.checkpoint(
                lambda t, layer = layer: layer(t, **kwargs), h, use_reentrant = True
            )
        else:
            h = layer(h, **kwargs)
    h.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    return list(calls)


def test_auto_turns_graphs_on_for_fixed_shape_launch_bound(monkeypatch):
    layers, calls, compiles, timed = _harness(monkeypatch)
    assert compiles == ["plain"]  # the graphed region is compiled only once graphs turn on
    for _ in range(bc._AUTO_WARMUP_PASSES):
        assert _step(layers, calls) == ["plain", "plain"]
    assert _step(layers, calls) == ["plain", "plain"]  # pass 3: timed, still graphs-off
    assert len(timed) == 2
    _step(layers, calls)  # pass 4 times the remaining calls, then decides
    assert bc._CUDAGRAPH_STATE["auto"]["decision"] == "on"
    assert _step(layers, calls) == ["graphs", "graphs"]
    assert compiles == ["plain", "graphs"]
    assert bc._CUDAGRAPH_STATE["auto"]["phase"] == "on"
    assert _step(layers, calls) == ["graphs", "graphs"]
    assert len(timed) == bc._AUTO_MEASURE_CALLS


def test_auto_keeps_graphs_off_when_gpu_bound(monkeypatch):
    layers, calls, compiles, _ = _harness(monkeypatch, host = 1e-3, gpu = 2e-3)
    for _ in range(8):
        assert _step(layers, calls) == ["plain", "plain"]
    auto = bc._CUDAGRAPH_STATE["auto"]
    assert auto["phase"] == "off" and auto["why"].startswith("GPU-bound")
    assert compiles == ["plain"]


def test_auto_ratio_threshold(monkeypatch):
    """Launching must be clearly slower than running: A100 ratios 0.55-0.92 gained nothing with graphs
    forced on (-1.2% to +1.0%), 1.26 gained 12.6%."""
    layers, calls, _, _ = _harness(monkeypatch, host = 1.1e-3, gpu = 1e-3)
    for _ in range(6):
        assert _step(layers, calls) == ["plain", "plain"]
    assert bc._CUDAGRAPH_STATE["auto"]["phase"] == "off"


def test_auto_shape_change_while_probing_keeps_graphs_off(monkeypatch):
    layers, calls, _, timed = _harness(monkeypatch)
    _step(layers, calls, L = 16)
    _step(layers, calls, L = 24)  # padded to the longest row: variable lengths
    auto = bc._CUDAGRAPH_STATE["auto"]
    assert auto["phase"] == "off" and auto["why"] == "batch shape changed"
    for _ in range(6):
        assert _step(layers, calls, L = 16) == ["plain", "plain"]
    assert not timed


def test_auto_after_on_new_shapes_past_the_cap_turn_graphs_off(monkeypatch):
    monkeypatch.setattr(bc, "_CUDAGRAPH_MAX_SHAPES", 2)
    released = []
    monkeypatch.setattr(bc, "_release_graph_pool", lambda: released.append(1))
    layers, calls, _, _ = _harness(monkeypatch)
    for _ in range(5):
        _step(layers, calls)
    assert _step(layers, calls) == ["graphs", "graphs"]
    assert _step(layers, calls, L = 8) == ["graphs", "graphs"]  # short last batch: second shape
    assert _step(layers, calls, L = 24) == ["plain", "plain"]  # third: over the cap
    assert bc._CUDAGRAPH_STATE["off"] == "shapes"
    assert released
    assert _step(layers, calls) == ["plain", "plain"]


def test_auto_graphed_compile_owns_only_its_compile_ids(monkeypatch):
    # The graphed region auto mode compiles once graphs turn on must use the owning backend too:
    # a foreign reduce-overhead function first called inside the block is not ours.
    from torch._guards import CompileContext, CompileId, compile_context

    layers, calls, compiles, _ = _harness(monkeypatch)
    monkeypatch.setattr(
        torch._TorchCompileInductorWrapper, "__call__", lambda self, gm, inputs, **kw: gm
    )
    manager = types.SimpleNamespace(ids_to_funcs = {}, id_to_compile_id = {})
    monkeypatch.setattr(bc, "_tree_manager", lambda device: manager)
    block_cid = CompileId(frame_id = 1, frame_compile_id = 0)
    foreign_cid = CompileId(frame_id = 2, frame_compile_id = 0)

    def register(cid):
        i = len(manager.ids_to_funcs)
        manager.ids_to_funcs[i] = None
        manager.id_to_compile_id[i] = cid

    def fake_compile(fn, **kwargs):
        compiles.append("graphs")
        backend = kwargs.get("backend")

        def run(*args, **kw):
            calls.append("graphs")
            if not manager.ids_to_funcs:  # first call records the block's tree function
                with compile_context(CompileContext(block_cid)):
                    if backend is not None:
                        backend("gm", [])
                register(block_cid)
                register(foreign_cid)  # foreign region behind a graph break, first call
            return fn(*args, **kw)

        return run

    monkeypatch.setattr(torch, "compile", fake_compile)
    for _ in range(4):
        assert "graphs" not in _step(layers, calls)
    assert bc._CUDAGRAPH_STATE["auto"]["decision"] == "on"
    assert not manager.ids_to_funcs
    assert _step(layers, calls) == ["graphs", "graphs"]
    assert compiles == ["plain", "graphs"]
    assert bc._CUDAGRAPH_STATE["ours"] == {block_cid}
    assert bc._pool_release_verdict([manager], bc._CUDAGRAPH_STATE["ours"]) == "foreign"


def test_auto_packed_rows_keep_graphs_off(monkeypatch):
    layers, calls, _, timed = _harness(monkeypatch)
    lengths = torch.tensor([8, 8])
    for _ in range(6):
        assert _step(layers, calls, packed_seq_lengths = lengths) == ["plain", "plain"]
    auto = bc._CUDAGRAPH_STATE["auto"]
    assert auto["phase"] == "off" and "packed" in auto["why"]
    assert not timed


def test_auto_low_free_memory_keeps_graphs_off(monkeypatch):
    layers, calls, _, _ = _harness(monkeypatch, free_gb = 1.0)
    for _ in range(6):
        assert _step(layers, calls) == ["plain", "plain"]
    assert bc._CUDAGRAPH_STATE["auto"]["why"].endswith("GiB free")


def test_auto_static_reason_keeps_graphs_off(monkeypatch):
    layers, calls, _, timed = _harness(monkeypatch, static = "distributed")
    for _ in range(6):
        assert _step(layers, calls) == ["plain", "plain"]
    assert bc._CUDAGRAPH_STATE["auto"]["why"] == "distributed"
    assert not timed


def test_auto_counts_passes_not_recomputes(monkeypatch):
    """Reentrant checkpointing calls each block again inside backward: those calls are not forward
    passes, so graphs still start after the same number of steps, at a pass boundary."""
    layers, calls, _, timed = _harness(monkeypatch)
    for _ in range(bc._AUTO_WARMUP_PASSES):
        # the first step also runs each block's grad-mode trial before its first no-grad call
        assert set(_step(layers, calls, checkpoint = True)) == {"plain"}
    _step(layers, calls, checkpoint = True)
    _step(layers, calls, checkpoint = True)
    assert len(timed) == bc._AUTO_MEASURE_CALLS  # forward calls only, never a recompute
    assert _step(layers, calls, checkpoint = True) == ["graphs"] * 4


def test_explicit_off_never_probes(monkeypatch):
    monkeypatch.setattr(bc, "_CUDAGRAPH_MODE", "off")
    layers, calls, compiles, timed = _harness(monkeypatch)
    for _ in range(6):
        assert _step(layers, calls) == ["plain", "plain"]
    assert not timed and compiles == ["plain"]
    assert bc._CUDAGRAPH_STATE["auto"]["phase"] == "probe"  # untouched


def test_static_reason_rules():
    layer = torch.nn.Linear(4, 4)
    assert bc._auto_static_reason(layer, torch.zeros(1)) == "not a CUDA tensor"


def test_static_reason_full_finetuning(monkeypatch):
    class FakeCuda:
        is_cuda = True

    layer = torch.nn.Linear(4, 4)  # every weight trainable
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.14.1")
    assert bc._auto_static_reason(layer, FakeCuda()) == "full finetuning"
    for p in layer.parameters():
        p.requires_grad_(False)
    assert bc._auto_static_reason(layer, FakeCuda()) is None
    monkeypatch.setattr(torch, "__version__", "2.12.1+cu128")
    assert bc._auto_static_reason(layer, FakeCuda()) == "torch < 2.13"


def test_a_new_model_decides_again(monkeypatch):
    """Graphs on for one (launch-bound) model must not carry over to the next model loaded."""
    layers, calls, _, _ = _harness(monkeypatch)
    for _ in range(5):
        _step(layers, calls)
    assert _step(layers, calls) == ["graphs", "graphs"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bc, "_torch_supported", lambda: True)

    class NextDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([NextDecoderLayer(), NextDecoderLayer()])
    assert bc.patch_block_compile(model) == ["NextDecoderLayer"]
    assert bc._CUDAGRAPH_STATE["auto"]["phase"] == "probe"
    model.train()
    assert _step(model.layers, calls) == ["plain", "plain"]


@pytest.mark.parametrize(
    "version, hip, hooked",
    [("2.14.1", None, True), ("2.12.1+cu128", None, False), ("2.14.1", "7.0", False)],
)
def test_auto_hooks_the_model_only_where_graphs_can_turn_on(monkeypatch, version, hip, hooked):
    """The hidden-states watcher only serves graphs: no forward hook where auto can never use them."""
    monkeypatch.setattr(bc, "_CUDAGRAPH_MODE", "auto")
    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bc, "_torch_supported", lambda: True)

    class HookDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states)

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([HookDecoderLayer(), HookDecoderLayer()])
    assert bc.patch_block_compile(model) == ["HookDecoderLayer"]
    assert model.__dict__.get("_unsloth_watch_hidden_states", False) is hooked


def test_probe_times_on_the_blocks_device(monkeypatch):
    """A block on cuda:1 while cuda:0 is current: sleep and both events must run on cuda:1."""
    current = ["cuda:0"]
    log = []

    class Guard:
        def __init__(self, device):
            self.device = str(device)

        def __enter__(self):
            self.prev, current[0] = current[0], self.device

        def __exit__(self, *exc):
            current[0] = self.prev

    class Event:
        def __init__(self, enable_timing = False):
            pass

        def record(self):
            log.append(("record", current[0]))

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 2.0

    monkeypatch.setattr(torch.cuda, "device", Guard)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device = None: None)
    monkeypatch.setattr(torch.cuda, "_sleep", lambda cycles: log.append(("sleep", current[0])))
    out, host, gpu = bc._auto_time(
        lambda: log.append(("call", current[0])) or "out", torch.device("cuda", 1)
    )
    assert out == "out" and gpu == 2.0 / 1e3 and host is not None
    assert log == [
        ("sleep", "cuda:1"),
        ("record", "cuda:1"),
        ("call", "cuda:1"),
        ("record", "cuda:1"),
    ]
    assert current[0] == "cuda:0"


def test_auto_blocks_on_several_devices_keep_graphs_off(monkeypatch):
    # device_map split: the first pass already visits both shards, before any timing or memory read.
    layers, calls, compiles, timed = _harness(monkeypatch)
    shard = {id(layers[0]): "cuda:0", id(layers[1]): "cuda:1"}
    current = []
    for layer in layers:
        layer.register_forward_pre_hook(lambda m, a, layer = layer: current.append(shard[id(layer)]))
    monkeypatch.setattr(bc, "_auto_block_device", lambda hs: current[-1], raising = False)
    for _ in range(6):
        assert _step(layers, calls) == ["plain", "plain"]
    auto = bc._CUDAGRAPH_STATE["auto"]
    assert auto["phase"] == "off" and auto["why"] == "blocks on several devices"
    assert compiles == ["plain"] and not timed
