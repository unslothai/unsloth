# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whole-model CPU offload that points each weight back at its host tensor instead of copying it.

CPU tests onload to the ``meta`` device: a kept weight comes back to its original host storage, and
a weight the stock path would have to copy raises (meta has no data), which is how a test tells the
two paths apart. The GPU test runs real accelerate hooks and checks bit-identical output.
"""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")
hooks = pytest.importorskip("accelerate.hooks")

from core.inference import diffusion_memory as dm  # noqa: E402


def _pipe(device, *names):
    """A pipeline shaped like diffusers': components, and an enable_model_cpu_offload that (re)builds
    one chained CpuOffload hook per component, as diffusers does after every call."""
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
    # Idempotent: a second call wraps nothing new.
    assert dm.keep_cpu_weights_on_offload(pipe) == 0


def test_hooks_rebuilt_by_a_later_enable_are_wrapped_too():
    pipe = _pipe("meta", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    for _ in range(3):
        # diffusers' maybe_free_model_hooks re-runs enable_model_cpu_offload after every call.
        pipe.enable_model_cpu_offload()
        assert _wrapped(pipe.transformer)


def test_a_device_that_hands_back_new_parameters_takes_the_stock_path():
    # meta tensors are not shallow-copy compatible with host ones, so the move makes new Parameter objects and
    # the offload must fall back to the stock copy (which raises on meta: there is no data to copy).
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
    out = pipe.transformer(pipe.text_encoder(x)).detach().cpu()
    pipe.transformer._hf_hook.init_hook(pipe.transformer)
    pipe.enable_model_cpu_offload()
    return out


@cuda
def test_real_offload_keeps_host_storage_and_is_bit_identical():
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
        # Offloaded through the kept host tensors: the very same storage every time.
        assert {n: _ptrs(m) for n, m in pipe.components.items()} == host


@cuda
def test_a_weight_written_on_the_device_is_copied_back():
    pipe = _pipe("cuda", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    before = pipe.transformer.weight.detach().clone()
    pipe.transformer(torch.ones(1, 8))
    with torch.no_grad():
        pipe.transformer.weight.add_(1)  # e.g. a LoRA fused on the device
    pipe.transformer._hf_hook.init_hook(pipe.transformer)
    assert pipe.transformer.weight.device.type == "cpu"
    assert torch.equal(pipe.transformer.weight.detach(), before + 1)


@cuda
def test_a_parameter_replaced_on_the_device_is_copied_back():
    pipe = _pipe("cuda", "transformer")
    pipe.enable_model_cpu_offload()
    dm.keep_cpu_weights_on_offload(pipe)
    pipe.transformer(torch.ones(1, 8))
    # A new Parameter under the same name can sit at the very version recorded at onload, so only its
    # identity tells it apart.
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
