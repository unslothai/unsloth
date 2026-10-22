# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whole-component offload onloads pinned weights one chunk at a time, not one tensor at a time.

The offload chain empties the device cache between components, so a per-tensor onload pays one cudaMalloc per
weight (and one cudaFree on the way out). Under host load that ran at 57-120 MiB/s instead of the link's 50 GiB/s.
CPU tests fake the pinning and onload to ``meta``: a per-chunk onload leaves every weight a view of one shared
allocation, a per-tensor onload gives every weight its own.
"""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")
hooks = pytest.importorskip("accelerate.hooks")

from core.inference import diffusion_memory as dm  # noqa: E402

N_LAYERS = 6


def _stack():
    torch.manual_seed(0)
    return torch.nn.Sequential(*(torch.nn.Linear(16, 16) for _ in range(N_LAYERS)))


def _pipe(device, *names):
    comps = {name: _stack() for name in names}
    pipe = types.SimpleNamespace(components = comps)

    def enable_model_cpu_offload(device = device):
        prev = None
        for module in comps.values():
            hooks.remove_hook_from_module(module)
            module.to("cpu")
            hook = hooks.CpuOffload(execution_device = device, prev_module_hook = prev)
            hooks.add_hook_to_module(module, hook)
            prev = hooks.UserCpuOffloadHook(module, hook)

    pipe.enable_model_cpu_offload = enable_model_cpu_offload
    for name, module in comps.items():
        setattr(pipe, name, module)
    return pipe


@pytest.fixture
def fake_pin(monkeypatch):
    """Pinning without CUDA: plain host chunks, small enough that one module spans several."""
    real_empty = torch.empty
    device_allocs: list = []

    def empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        out = real_empty(*args, **kwargs)
        if out.device.type != "cpu":
            device_allocs.append(out.nbytes)
        return out

    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    monkeypatch.setattr(dm, "_PIN_CHUNK_BYTES", 4096)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "empty", empty)
    return device_allocs


def _storages(module):
    return {p.untyped_storage().nbytes() for p in module.parameters()}


def _onload(pipe):
    module = pipe.transformer
    module._hf_hook.pre_forward(module, torch.zeros(1, 16))
    return module


def test_pinning_records_the_chunk_layout(fake_pin):
    module = _stack()
    host = {n: p.data for n, p in module.named_parameters()}
    layout: list = []
    assert dm._pin_host_weights(module, host, layout = layout) > 0
    assert 1 < len(layout) < len(host)
    names = [name for _, entries in layout for name, _, _ in entries]
    assert sorted(names) == sorted(host)
    for buf, entries in layout:
        for name, start, view in entries:
            assert view is host[name]
            assert view.data_ptr() == buf.data_ptr() + start


def test_a_device_that_cannot_back_host_parameters_takes_the_stock_copy(fake_pin):
    # meta tensors cannot be set as a CPU Parameter's data; the stock move hands back new Parameters there.
    pipe = _pipe("meta", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    module = _onload(pipe)
    assert module.__dict__[dm._KEEP_ATTR]["pin_layout"]
    assert all(p.device.type == "meta" for p in module.parameters())
    assert all(p.untyped_storage().nbytes() == p.nbytes for p in module.parameters())


@pytest.mark.parametrize(
    "value,on", [("", True), ("1", True), ("0", False), ("off", False), ("False", False)]
)
def test_fast_upload_switch(monkeypatch, value, on):
    monkeypatch.setenv(dm.OFFLOAD_FAST_UPLOAD_ENV, value)
    assert dm._fast_upload_enabled() is on


def _fake_pipeline_utils(monkeypatch):
    calls: list = []
    module = types.ModuleType("diffusers.pipelines.pipeline_utils")
    module.empty_device_cache = lambda *a, **k: calls.append(a)
    monkeypatch.setitem(sys.modules, "diffusers.pipelines.pipeline_utils", module)
    return module, calls


def test_the_re_enable_keeps_the_device_cache_only_once_chunks_are_in_use(monkeypatch):
    module, calls = _fake_pipeline_utils(monkeypatch)
    lin = torch.nn.Linear(2, 2)
    pipe = types.SimpleNamespace(components = {"transformer": lin})
    with dm._keep_device_cache(pipe):
        module.empty_device_cache("cuda")
    assert calls == [("cuda",)]  # nothing pinned yet: the stock empty_cache runs
    lin.__dict__[dm._KEEP_ATTR] = {"pin_layout": [object()]}
    with dm._keep_device_cache(pipe):
        module.empty_device_cache("cuda")
    assert calls == [("cuda",)]
    module.empty_device_cache("cuda")  # restored on exit
    assert len(calls) == 2
    monkeypatch.setenv(dm.OFFLOAD_FAST_UPLOAD_ENV, "0")
    with dm._keep_device_cache(pipe):
        module.empty_device_cache("cuda")
    assert len(calls) == 3


def test_the_device_cache_is_restored_when_the_re_enable_raises(monkeypatch):
    module, calls = _fake_pipeline_utils(monkeypatch)
    original = module.empty_device_cache
    lin = torch.nn.Linear(2, 2)
    lin.__dict__[dm._KEEP_ATTR] = {"pin_layout": [object()]}
    with pytest.raises(RuntimeError):
        with dm._keep_device_cache(types.SimpleNamespace(components = {"t": lin})):
            raise RuntimeError("boom")
    assert module.empty_device_cache is original


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


def _render(
    pipe,
    x,
    ctx = torch.no_grad,
):
    out = x
    with ctx():
        for module in pipe.components.values():
            out = module(out.to(next(module.parameters()).device))
    out = out.detach().cpu()
    pipe.transformer._hf_hook.init_hook(pipe.transformer)
    pipe.enable_model_cpu_offload()
    return out


def _device_allocs():
    return torch.cuda.memory_stats().get("num_device_alloc", 0)


@cuda
@pytest.mark.parametrize("mode", ["no_grad", "inference_mode"])
def test_chunk_onload_is_bit_identical_and_stops_allocating(monkeypatch, mode):
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    monkeypatch.setattr(dm, "_PIN_CHUNK_BYTES", 4096)
    ctx = torch.inference_mode if mode == "inference_mode" else torch.no_grad
    x = torch.randn(4, 16)
    stock = _pipe("cuda", "text_encoder", "transformer")
    stock.enable_model_cpu_offload()
    ref = _render(stock, x, ctx)
    pipe = _pipe("cuda", "text_encoder", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    assert torch.equal(_render(pipe, x, ctx), ref)  # the first render pins
    layouts = {n: m.__dict__[dm._KEEP_ATTR]["pin_layout"] for n, m in pipe.components.items()}
    assert all(1 < len(layout) < N_LAYERS * 2 for layout in layouts.values())
    seen = []

    def spy(module, *args):
        # A post-hook: accelerate onloads inside forward, after any pre-hook.
        assert all(p.device.type == "cuda" for p in module.parameters())
        seen.append({p.untyped_storage().data_ptr() for p in module.parameters()})

    for module in pipe.components.values():
        module.register_forward_hook(spy)
    allocs = _device_allocs()
    for _ in range(3):
        assert torch.equal(_render(pipe, x, ctx), ref)
    # Onloaded weights are views of one allocation per chunk (per tensor would be one storage per weight) ...
    assert [len(s) for s in seen] == [len(layouts["text_encoder"]), len(layouts["transformer"])] * 3
    # ... and the chunks come back out of the caching allocator: no cudaMalloc per swap.
    assert _device_allocs() - allocs <= 2
    host = pipe.transformer.__dict__[dm._KEEP_ATTR]["host"]
    assert all(
        p.data_ptr() == host[n].data_ptr() and p.is_pinned()
        for n, p in pipe.transformer.named_parameters()
    )


@cuda
def test_kill_switch_restores_the_per_tensor_onload(monkeypatch):
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    monkeypatch.setenv(dm.OFFLOAD_FAST_UPLOAD_ENV, "0")
    pipe = _pipe("cuda", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    module = pipe.transformer
    module._hf_hook.pre_forward(module, torch.zeros(1, 16))
    assert all(p.device.type == "cuda" for p in module.parameters())
    assert all(p.untyped_storage().nbytes() == p.nbytes for p in module.parameters())


@cuda
def test_a_weight_replaced_while_offloaded_takes_the_stock_copy(monkeypatch):
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    monkeypatch.setattr(dm, "_PIN_CHUNK_BYTES", 4096)
    pipe = _pipe("cuda", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    module = pipe.transformer
    _render(pipe, torch.randn(4, 16))
    fresh = torch.nn.Parameter(torch.full((16, 16), 0.5))
    module[0].weight = fresh
    module._hf_hook.pre_forward(module, torch.zeros(1, 16))
    assert all(p.device.type == "cuda" for p in module.parameters())
    assert module[0].weight.untyped_storage().nbytes() == fresh.nbytes
    assert torch.equal(module[0].weight.cpu(), torch.full((16, 16), 0.5))
    assert module[1].weight.untyped_storage().nbytes() > module[1].weight.nbytes


@cuda
def test_a_weight_written_on_the_device_after_a_chunk_onload_is_copied_back(monkeypatch):
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    pipe = _pipe("cuda", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    module = pipe.transformer
    _render(pipe, torch.randn(4, 16))
    before = module[2].weight.detach().clone()
    module._hf_hook.pre_forward(module, torch.zeros(1, 16))
    with torch.no_grad():
        module[2].weight.add_(1)
    module._hf_hook.init_hook(module)
    assert module[2].weight.device.type == "cpu"
    assert torch.equal(module[2].weight.detach(), before + 1)
