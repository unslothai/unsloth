# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU tests onload to ``meta``: a kept weight returns to its host storage, a stock copy raises (no data)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
hooks = pytest.importorskip("accelerate.hooks")

from core.inference import diffusion_memory as dm  # noqa: E402


def _pipe(device, *names):
    """Diffusers-shaped pipe whose enable_model_cpu_offload rebuilds chained CpuOffload hooks each call."""
    torch.manual_seed(0)
    comps = {name: torch.nn.Linear(8, 8) for name in names}
    pipe = types.SimpleNamespace(components = comps, enabled = 0)

    def enable_model_cpu_offload(device = device):
        pipe.enabled += 1
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


def _ptrs(module):
    return [p.data_ptr() for p in module.parameters()]


def _wrapped(module):
    return bool(getattr(module._hf_hook, dm._KEEP_ATTR, False))


def test_every_offload_hook_is_wrapped():
    pipe = _pipe("meta", "text_encoder", "transformer")
    pipe.enable_model_cpu_offload()
    assert dm.keep_cpu_weights_on_offload(pipe) == 2
    assert _wrapped(pipe.text_encoder) and _wrapped(pipe.transformer)
    assert dm.keep_cpu_weights_on_offload(pipe) == 0


def test_hooks_rebuilt_by_a_later_enable_are_wrapped_too():
    pipe = _pipe("meta", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    for _ in range(3):
        pipe.enable_model_cpu_offload()
        assert _wrapped(pipe.transformer)


def test_a_device_that_hands_back_new_parameters_takes_the_stock_path():
    pipe = _pipe("meta", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    pipe.transformer(torch.ones(1, 8))
    assert pipe.transformer.weight.device.type == "meta"
    with pytest.raises(NotImplementedError):
        pipe.transformer._hf_hook.init_hook(pipe.transformer)


def test_kill_switch_and_missing_hooks_leave_the_stock_path(monkeypatch):
    pipe = _pipe("meta", "transformer")
    pipe.enable_model_cpu_offload()
    monkeypatch.setenv(dm.OFFLOAD_KEEP_CPU_ENV, "0")
    assert dm.keep_cpu_weights_on_offload(pipe) == 0
    assert not _wrapped(pipe.transformer)
    monkeypatch.delenv(dm.OFFLOAD_KEEP_CPU_ENV)
    assert dm.keep_cpu_weights_on_offload(types.SimpleNamespace(components = {})) == 0


def test_subclass_weights_are_not_kept():
    class Sub(torch.Tensor):
        pass

    lin = torch.nn.Linear(2, 2)
    assert dm._keepable(lin.weight)
    lin.weight = torch.nn.Parameter(lin.weight.detach().as_subclass(Sub))
    assert not dm._keepable(lin.weight)


def test_model_offload_plan_wraps_the_pipeline():
    pipe = _pipe("meta", "transformer")
    plan = types.SimpleNamespace(
        offload_policy = dm.OFFLOAD_MODEL, vae_tiling = False, vae_slicing = False
    )
    policy, _ = dm.apply_memory_plan(pipe, plan, device = "meta")
    assert policy == dm.OFFLOAD_MODEL
    assert _wrapped(pipe.transformer)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


def _render(pipe, x):
    out = x
    for module in pipe.components.values():
        out = module(out)
    out = out.detach().cpu()
    pipe.transformer._hf_hook.init_hook(pipe.transformer)
    pipe.enable_model_cpu_offload()
    return out


@cuda
def test_real_offload_keeps_host_storage_and_is_bit_identical(monkeypatch):
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "0")
    x = torch.randn(4, 8)
    stock = _pipe("cuda", "text_encoder", "transformer")
    stock.enable_model_cpu_offload()
    ref = _render(stock, x)
    pipe = _pipe("cuda", "text_encoder", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    host = {n: _ptrs(m) for n, m in pipe.components.items()}
    for _ in range(3):
        assert torch.equal(_render(pipe, x), ref)
        assert {n: _ptrs(m) for n, m in pipe.components.items()} == host


@cuda
def test_a_weight_written_on_the_device_is_copied_back():
    pipe = _pipe("cuda", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    before = pipe.transformer.weight.detach().clone()
    pipe.transformer(torch.ones(1, 8))
    with torch.no_grad():
        pipe.transformer.weight.add_(1)
    pipe.transformer._hf_hook.init_hook(pipe.transformer)
    assert pipe.transformer.weight.device.type == "cpu"
    assert torch.equal(pipe.transformer.weight.detach(), before + 1)


@cuda
def test_weights_are_pinned_on_first_onload_into_shared_chunks(monkeypatch):
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    x = torch.randn(4, 8)
    stock = _pipe("cuda", "text_encoder", "transformer")
    stock.enable_model_cpu_offload()
    ref = _render(stock, x)
    pipe = _pipe("cuda", "text_encoder", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    assert not pipe.transformer.weight.is_pinned()  # nothing locked at load
    for _ in range(2):
        assert torch.equal(_render(pipe, x), ref)
    for module in pipe.components.values():
        assert all(p.is_pinned() for p in module.parameters())
        assert (
            module.weight.untyped_storage().data_ptr() == module.bias.untyped_storage().data_ptr()
        )


@cuda
def test_pinning_is_off_by_switch_and_when_ram_is_short(monkeypatch):
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "0")
    pipe = _pipe("cuda", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    _render(pipe, torch.randn(4, 8))
    assert not pipe.transformer.weight.is_pinned()

    psutil = pytest.importorskip("psutil")
    monkeypatch.delenv(dm.OFFLOAD_PIN_ENV)
    short = types.SimpleNamespace(total = 16 << 30, available = 2 << 30)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: short)
    pipe = _pipe("cuda", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    _render(pipe, torch.randn(4, 8))
    assert not pipe.transformer.weight.is_pinned()


@cuda
def test_a_non_contiguous_weight_keeps_its_layout(monkeypatch):
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    pipe = _pipe("cuda", "transformer")
    lin = pipe.transformer
    lin.weight = torch.nn.Parameter(lin.weight.detach().t().contiguous().t())  # transposed strides
    assert not lin.weight.is_contiguous()
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    _render(pipe, torch.randn(4, 8))
    assert not lin.weight.is_pinned() and lin.bias.is_pinned()
    assert lin.weight.stride() == (1, 8)


@cuda
def test_a_parameter_replaced_on_the_device_is_copied_back():
    pipe = _pipe("cuda", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    pipe.transformer(torch.ones(1, 8))
    # Same name and version as the onloaded weight: only identity tells it apart.
    old = pipe.transformer.weight
    new = torch.nn.Parameter(torch.full_like(old, 3.0))
    with torch.no_grad():
        while new._version < old._version:
            new.mul_(1)
    assert new._version == old._version
    pipe.transformer.weight = new
    pipe.transformer._hf_hook.init_hook(pipe.transformer)
    assert pipe.transformer.weight.device.type == "cpu"
    assert torch.equal(pipe.transformer.weight.detach(), torch.full((8, 8), 3.0))


def _host_of(module):
    return {name: p.data for name, p in module.named_parameters()}


@cuda
def test_the_ram_gate_counts_the_chunks_really_allocated(monkeypatch):
    psutil = pytest.importorskip("psutil")
    # Weight 256 B and bias 32 B (256 aligned) do not share a 384 B chunk: 384 + 256 B are allocated for 512.
    monkeypatch.setattr(dm, "_PIN_CHUNK_BYTES", 384)
    reserve = 4 << 30
    lin = torch.nn.Linear(8, 8)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(total = 16 << 30, available = reserve + 600),
    )
    assert dm._pin_host_weights(lin, _host_of(lin)) == 0
    assert not lin.weight.is_pinned()
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(total = 16 << 30, available = reserve + 640),
    )
    assert dm._pin_host_weights(lin, _host_of(lin)) == 640
    assert lin.weight.is_pinned() and lin.bias.is_pinned()


@cuda
def test_a_failed_pin_hands_the_partial_chunks_back(monkeypatch):
    monkeypatch.setenv(dm.OFFLOAD_PIN_ENV, "1")
    monkeypatch.setattr(dm, "_PIN_CHUNK_BYTES", 384)
    real_empty, calls, emptied = torch.empty, [], []

    def empty(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("cudaHostAlloc failed")
        return real_empty(*args, **kwargs)

    lin = torch.nn.Linear(8, 8)
    host = _host_of(lin)
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(dm, "_host_empty_cache", lambda: emptied.append(True))
    assert dm._pin_host_weights(lin, host) == 0
    assert len(calls) == 2 and emptied
    assert not lin.weight.is_pinned() and not lin.bias.is_pinned()


@cuda
@pytest.mark.parametrize("how", ["parameter", "data"])
def test_a_parameter_replaced_while_offloaded_is_not_restored_to_the_old_weight(how):
    pipe = _pipe("cuda", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    pipe.transformer(torch.ones(1, 8))
    pipe.transformer._hf_hook.init_hook(pipe.transformer)
    assert pipe.transformer.weight.device.type == "cpu"
    fresh = torch.full((8, 8), 5.0)
    if how == "parameter":
        pipe.transformer.weight = torch.nn.Parameter(fresh)
    else:
        pipe.transformer.weight.data = fresh
    out = pipe.transformer(torch.ones(1, 8))
    assert torch.allclose(
        out.cpu(), torch.full((1, 8), 40.0) + pipe.transformer.bias.detach().cpu()
    )
    pipe.transformer._hf_hook.init_hook(pipe.transformer)
    assert pipe.transformer.weight.device.type == "cpu"
    assert torch.equal(pipe.transformer.weight.detach(), torch.full((8, 8), 5.0))


@cuda
def test_a_weight_reassigned_through_data_on_the_device_is_copied_back():
    pipe = _pipe("cuda", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    pipe.transformer(torch.ones(1, 8))
    version = pipe.transformer.weight._version
    pipe.transformer.weight.data = torch.full((8, 8), 7.0, device = "cuda")
    assert pipe.transformer.weight._version == version
    pipe.transformer._hf_hook.init_hook(pipe.transformer)
    assert pipe.transformer.weight.device.type == "cpu"
    assert torch.equal(pipe.transformer.weight.detach(), torch.full((8, 8), 7.0))
