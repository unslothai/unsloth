# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A portable (other layout) optimizer resume under UNSLOTH_OFFLOAD_OPTIMIZER_STATE: the loaded state ends on the
host and never sits on the device all at once, for a torch AdamW and a bitsandbytes AdamW8bit alike."""

import importlib.util
import os
import sys

import pytest
import torch
from real_accelerator import has_real_cuda

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

pytestmark = pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")


def _state_offload():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_state_offload", os.path.join(_ROOT, "unsloth", "optimizers", "state_offload.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _portable():
    sys.path.insert(0, os.path.join(_ROOT, "tests", "_shared"))
    from unsloth_models_loader import load_models_module

    module = load_models_module("portable_resume")
    assert module.__file__.startswith(os.path.join(_ROOT, "unsloth")), module.__file__
    return module


def _model(n = 4):
    torch.manual_seed(0)
    # n 1M-parameter weights: their two float32 Adam moments are 8 MiB each.
    return torch.nn.Sequential(*[torch.nn.Linear(1024, 1024, bias = False) for _ in range(n)]).cuda()


def _saved_state(portable, model):
    """The 1-GPU checkpoint: torch AdamW moments, name keyed, read with map_location = "cpu"."""
    source = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    for p in model.parameters():
        p.grad = torch.randn_like(p)
    source.step()
    named = portable.as_named_state(model, source, source.state_dict())
    cpu = lambda v: v.cpu().clone() if torch.is_tensor(v) else v
    return {
        "state": {n: {k: cpu(v) for k, v in e.items()} for n, e in named["state"].items()},
        "param_groups": named["param_groups"],
    }


def _state_devices(optimizer):
    return {
        t.device.type
        for s in optimizer.state.values()
        for t in s.values()
        if torch.is_tensor(t) and t.numel() > 1
    }


def _load(portable, model, optimizer, saved):
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    portable.portable_load_optimizer(model, optimizer, saved)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base, torch.cuda.memory_allocated() - base


def test_portable_resume_keeps_an_offloaded_torch_state_on_the_host():
    portable, offload = _portable(), _state_offload()
    model = _model()
    saved = _saved_state(portable, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    assert offload.offload_optimizer_state(optimizer, chunk_bytes = 9 << 20) is not None
    peak, held = _load(portable, model, optimizer, saved)
    assert _state_devices(optimizer) == {"cpu"}
    # Every moment moved to the device first would be 32 MiB at once.
    assert peak < 1 << 20 and held < 1 << 20, (peak, held)
    name = next(iter(saved["state"]))
    p = dict(model.named_parameters())[name]
    assert torch.equal(optimizer.state[p]["exp_avg"].cpu(), saved["state"][name]["exp_avg"])
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    assert _state_devices(optimizer) == {"cpu"}


def test_portable_resume_offloads_a_bitsandbytes_state():
    bnb = pytest.importorskip("bitsandbytes")
    portable, offload = _portable(), _state_offload()
    model = _model(16)
    saved = _saved_state(portable, model)
    for p in model.parameters():
        p.grad = None
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr = 1e-3)
    assert offload.offload_optimizer_state(optimizer, chunk_bytes = 3 << 20) is not None
    peak, held = _load(portable, model, optimizer, saved)
    # The 8-bit state is offloaded right after the import, not left on the device until the first step.
    assert _state_devices(optimizer) == {"cpu"}
    assert held < 1 << 20, held
    # Each float32 moment reaches the device only chunk by chunk as it is quantized: the peak is the 8-bit
    # moments (16 MiB) and one chunk's quantization, not the 128 MiB of float32 moments at once.
    assert peak < 128 << 20, peak
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    optimizer.step()
    # The first step stays chunked: one parameter's 8-bit moments (about 2 MiB) at a time, not all 16 (32 MiB).
    assert torch.cuda.max_memory_allocated() - base < 8 << 20
    assert _state_devices(optimizer) == {"cpu"}
