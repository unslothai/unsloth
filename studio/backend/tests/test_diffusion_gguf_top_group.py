# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A streamed GGUF denoiser's top-level group (embedders, modulation, final projection: about 1.1 GiB at float32 on
FLUX.3) re-points to its host copy on offload instead of copying it device to host every step."""

from __future__ import annotations

import sys
import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_memory as mem  # noqa: E402


class GGUFParameter(torch.nn.Parameter):
    """Stand-in for GGUFParameter; importing the real one would leak via ``sys.modules`` into other tests."""

    def __new__(
        cls,
        data,
        requires_grad = False,
        quant_type = None,
    ):
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.quant_type = quant_type
        return self


Q8_0 = 8
# Literal, not mem.TOP_GROUP_GGUF_ENV: the behavioural tests must also run (and fail) on a tree without the name.
TOP_GROUP_GGUF_ENV = "UNSLOTH_DIFFUSION_TOP_GROUP_GGUF"


@pytest.fixture(autouse = True)
def _fake_gguf_utils(monkeypatch):
    mod = types.ModuleType("diffusers.quantizers.gguf.utils")
    mod.GGUFParameter = GGUFParameter
    monkeypatch.setitem(sys.modules, "diffusers.quantizers.gguf.utils", mod)


def _gguf_weight():
    return GGUFParameter(torch.arange(4 * 34, dtype = torch.uint8).reshape(4, 34), quant_type = Q8_0)


def _top_group(
    monkeypatch,
    *,
    gguf_weight = True,
    other_subclass = False,
):
    go = types.ModuleType("diffusers.hooks.group_offloading")
    go._GROUP_OFFLOADING = "group_offloading"
    go._is_torchao_tensor = lambda t: False
    hooks = types.ModuleType("diffusers.hooks")
    hooks.group_offloading = go
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    monkeypatch.setitem(sys.modules, "diffusers.hooks.group_offloading", go)
    leaf = torch.nn.Linear(32, 4)
    if gguf_weight:
        leaf.weight = _gguf_weight()
    if other_subclass:

        class Other(torch.nn.Parameter):
            pass

        leaf.bias = Other(torch.zeros(4), requires_grad = False)
    group = types.SimpleNamespace(
        stream = None,
        offload_to_disk_path = None,
        onload_device = types.SimpleNamespace(type = "cuda"),
        modules = [leaf],
        parameters = [],
        buffers = [],
        onload_ = "diffusers",
        offload_ = "diffusers",
    )
    hook = types.SimpleNamespace(group = group)
    module = types.SimpleNamespace(
        _diffusers_hook = types.SimpleNamespace(
            get_hook = lambda name: hook if name == "group_offloading" else None
        )
    )
    return module, group, leaf


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    for name in (TOP_GROUP_GGUF_ENV, mem.PIN_TOP_GROUP_ENV, mem.GROUP_OFFLOAD_PIN_ENV):
        monkeypatch.delenv(name, raising = False)


def test_gguf_top_group_skips_the_copy_back(monkeypatch):
    module, group, leaf = _top_group(monkeypatch)
    weight = leaf.weight
    host_ptr = weight.data_ptr()
    assert mem._skip_top_level_copy_back(module) is True
    assert group._unsloth_top_no_copy_back is True
    host = getattr(group, mem.TOP_GROUP_HOST_ATTR)
    assert any(type(t) is GGUFParameter for t in host)
    group.offload_()
    # the Parameter object (and its quant type) is the one the forward reads; only its data is re-pointed
    assert leaf.weight is weight and type(leaf.weight) is GGUFParameter
    assert leaf.weight.quant_type == Q8_0
    assert leaf.weight.data_ptr() == host_ptr


def test_gguf_top_group_pins_when_the_budget_allows(monkeypatch):
    module, group, leaf = _top_group(monkeypatch)
    monkeypatch.setattr(mem, "host_pinning_allowed", lambda device = None: True)
    monkeypatch.setattr(mem, "_pinned_memory_capped", lambda: False)
    monkeypatch.setattr(mem, "_pin_budget_mib", lambda: 1 << 20)
    pinned = []
    real_pin = torch.Tensor.pin_memory
    monkeypatch.setattr(
        torch.Tensor, "pin_memory", lambda self, *a, **k: pinned.append(type(self)) or self
    )
    try:
        assert mem._pin_top_level_group(module) is True
    finally:
        monkeypatch.setattr(torch.Tensor, "pin_memory", real_pin)
    assert group._unsloth_pinned_top is True and len(pinned) == 2  # weight + bias
    assert type(leaf.weight) is GGUFParameter and leaf.weight.quant_type == Q8_0


def test_kill_switch_keeps_diffusers_path_for_gguf(monkeypatch):
    monkeypatch.setenv(TOP_GROUP_GGUF_ENV, "0")
    module, group, _ = _top_group(monkeypatch)
    assert mem._skip_top_level_copy_back(module) is False
    assert group.onload_ == "diffusers" and group.offload_ == "diffusers"
    module, group, _ = _top_group(monkeypatch, gguf_weight = False)
    assert mem._skip_top_level_copy_back(module) is True


def test_other_parameter_subclasses_still_keep_diffusers_path(monkeypatch):
    module, group, _ = _top_group(monkeypatch, other_subclass = True)
    assert mem._skip_top_level_copy_back(module) is False
    assert group.onload_ == "diffusers"


def test_the_kill_switch_name_matches_the_module():
    assert mem.TOP_GROUP_GGUF_ENV == TOP_GROUP_GGUF_ENV


def test_a_pinned_gguf_top_group_keeps_streamed_graphs_off(monkeypatch):
    """A pinned GGUF top group must still refuse streamed graph capture (VRAM for a 6 GiB T4 decode)."""
    import core.inference.diffusion_offload_prefetch as op

    module, group, _ = _top_group(monkeypatch)
    monkeypatch.setattr(mem, "host_pinning_allowed", lambda device = None: True)
    monkeypatch.setattr(mem, "_pinned_memory_capped", lambda: False)
    monkeypatch.setattr(mem, "_pin_budget_mib", lambda: 1 << 20)
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)
    assert mem._pin_top_level_group(module) is True
    assert group._unsloth_top_gguf is True
    monkeypatch.setattr(op, "module_prefetcher", lambda m: object())
    monkeypatch.setattr(mem, "_offload_groups", lambda m: [group])
    assert "GGUF top-level group" in (op.capture_refusal(module) or "")
    plain, plain_group, _ = _top_group(monkeypatch, gguf_weight = False)
    assert mem._pin_top_level_group(plain) is True and plain_group._unsloth_top_gguf is False
    monkeypatch.setattr(mem, "_offload_groups", lambda m: [plain_group])
    assert op.capture_refusal(plain) is None
