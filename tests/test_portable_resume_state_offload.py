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


def _fsdp2_world1(model, tmp_path):
    """FSDP2 over a one-rank NCCL world: every parameter is a DTensor (Shard(0)) on this GPU."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError:
        pytest.skip("torch without FSDP2 fully_shard")
    if not dist.is_nccl_available():
        pytest.skip("needs NCCL")
    dist.init_process_group(
        "nccl", store = dist.FileStore(str(tmp_path / "store"), 1), rank = 0, world_size = 1
    )
    mesh = init_device_mesh("cuda", (1,))
    for layer in model:
        fully_shard(layer, mesh = mesh)
    fully_shard(model, mesh = mesh)
    return model


def test_portable_resume_keeps_offloaded_dtensor_moments_on_the_host(tmp_path):
    """Offloaded DTensor moments load onto the host, not all onto the device at once."""
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    portable, offload = _portable(), _state_offload()
    plain = _model()
    saved = _saved_state(portable, plain)
    model = _model()
    try:
        model = _fsdp2_world1(model, tmp_path)
        optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3)
        assert offload.offload_optimizer_state(optimizer, chunk_bytes = 9 << 20) is not None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        portable._slice_maps(model)
        maps = torch.cuda.max_memory_allocated() - base
        peak, held = _load(portable, model, optimizer, saved)
        states = [s[k] for s in optimizer.state.values() for k in ("exp_avg", "exp_avg_sq")]
        assert len(states) == 8 and all(isinstance(v, DTensor) for v in states)
        assert {v.device.type for v in states} == {"cuda"}
        assert {v._local_tensor.device.type for v in states} == {"cpu"}
        # Peak stays at the 16 MiB slice map; device-first moments would be 32 MiB.
        assert peak <= maps + (1 << 20) and held < 1 << 20, (peak, maps, held)
        name = next(iter(saved["state"]))
        p = dict(model.named_parameters())[name]
        assert torch.equal(
            optimizer.state[p]["exp_avg"]._local_tensor, saved["state"][name]["exp_avg"]
        )
        # Matches one-GPU AdamW resumed from the same state.
        plain = _model()
        reference = torch.optim.AdamW(plain.parameters(), lr = 1e-3)
        reference.load_state_dict(
            {
                "state": {
                    i: saved["state"][n] for i, (n, _) in enumerate(plain.named_parameters())
                },
                "param_groups": [
                    {
                        **saved["param_groups"][0],
                        "params": list(range(len(list(plain.parameters())))),
                    }
                ],
            }
        )
        torch.manual_seed(1)
        for a, b in zip(model.parameters(), plain.parameters()):
            g = torch.randn(b.shape, device = "cuda")
            b.grad = g
            a.grad = DTensor.from_local(g, a.device_mesh, a.placements, run_check = False)
        optimizer.step()
        reference.step()
        for a, b in zip(model.parameters(), plain.parameters()):
            torch.testing.assert_close(a.full_tensor(), b.detach())
        assert {
            v._local_tensor.device.type
            for s in optimizer.state.values()
            for v in s.values()
            if isinstance(v, DTensor)
        } == {"cpu"}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
