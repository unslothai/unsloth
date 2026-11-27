# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_OFFLOAD_OPTIMIZER_STATE overlap on FSDP2 DTensor state: the staging ring copies each rank's
local shard, the state keeps its DTensor wrapper (host local shard between steps), and the updates are
bitwise the serial chunks'. Uneven and empty shards included. A fake 2-rank process group, one CUDA device."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch
from real_accelerator import has_real_cuda

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "unsloth", "optimizers", "state_offload.py")

_SCRIPT = textwrap.dedent(
    """
    import importlib.util, os, sys, torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard
    from torch.testing._internal.distributed.fake_pg import FakeStore

    dist.init_process_group("fake", store = FakeStore(), rank = 0, world_size = 2)
    spec = importlib.util.spec_from_file_location("_unsloth_state_offload", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    name = sys.argv[2]
    mesh = init_device_mesh("cuda", (2,))


    class ReplacingMomentum(torch.optim.Optimizer):
        def __init__(self, params, lr = 1e-2):
            super().__init__(params, dict(lr = lr))

        @torch.no_grad()
        def step(self, closure = None):
            if closure is not None:
                with torch.enable_grad():
                    closure()
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if "m" not in state:
                        state["m"] = torch.zeros_like(p)
                    state["m"] = state["m"] * 0.9 + p.grad
                    p.add_(state["m"], alpha = -group["lr"])


    ReplacingMomentum.__module__ = "torch.optim._unsloth_test"

    def make(kind, params):
        if kind == "adamw_foreach":
            return torch.optim.AdamW(params, lr = 1e-3, foreach = True, weight_decay = 0.01)
        if kind == "adamw_single":
            return torch.optim.AdamW(params, lr = 1e-3, foreach = False)
        if kind == "adamw_fused":
            return torch.optim.AdamW(params, lr = 1e-3, fused = True)
        if kind == "adam_amsgrad":
            return torch.optim.Adam(params, lr = 1e-3, amsgrad = True)
        return ReplacingMomentum(params)

    # This rank's local shards: uneven rows, and empty ones (rank 0 of 2 sees its share only).
    shapes = [(0, 7), (3, 5), (129, 64), (1,), (2048,), (65, 33), (0,)]

    def run(overlap):
        os.environ["UNSLOTH_OFFLOAD_OPTIMIZER_STATE_OVERLAP"] = overlap
        g = torch.Generator().manual_seed(0)
        params = [
            torch.nn.Parameter(DTensor.from_local(torch.randn(s, generator = g).cuda(), mesh, [Shard(0)], run_check = False))
            for s in shapes
        ]
        optimizer = make(name, params)
        offloader = module.offload_optimizer_state(optimizer, chunk_bytes = 24 << 10)
        for step in range(5):
            for p in params:
                local = torch.randn(p._local_tensor.shape, generator = g).cuda()
                p.grad = DTensor.from_local(local, mesh, [Shard(0)], run_check = False)
            optimizer.step()
            optimizer.zero_grad(set_to_none = True)
            for state in optimizer.state.values():
                for value in state.values():
                    if isinstance(value, torch.Tensor) and value.dim() > 0:
                        assert isinstance(value, DTensor), type(value)
                        assert value._local_tensor.device.type == "cpu", value._local_tensor.device
        torch.cuda.synchronize()
        weights = [p._local_tensor.detach().cpu().clone() for p in params]
        states = [
            (v._local_tensor if isinstance(v, DTensor) else v).cpu().clone()
            for p in params for k, v in sorted(optimizer.state[p].items()) if isinstance(v, torch.Tensor)
        ]
        held = optimizer.state_dict()["state"]
        assert all(isinstance(v, DTensor) for s in held.values() for v in s.values() if isinstance(v, torch.Tensor) and v.dim() > 0)
        return weights, states, offloader.overlapped_steps, len(offloader._chunks(params))

    serial = run("0")
    ring = run("1")
    assert serial[2] == 0, serial[2]
    assert serial[3] > 1, serial[3]
    for a, b in zip(serial[0] + serial[1], ring[0] + ring[1]):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(a.view(-1).view(torch.uint8) if a.numel() else a, b.view(-1).view(torch.uint8) if b.numel() else b)
    print("OVERLAPPED", ring[2], "CHUNKS", ring[3])
    dist.destroy_process_group()
    """
)


@pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")
@pytest.mark.parametrize(
    "name", ["adamw_foreach", "adamw_single", "adamw_fused", "adam_amsgrad", "replacing"]
)
def test_dtensor_state_overlap_matches_serial(tmp_path, name):
    pytest.importorskip("torch.distributed.tensor")
    try:
        from torch.testing._internal.distributed.fake_pg import FakeStore  # noqa: F401
    except Exception:
        pytest.skip("torch without the fake process group")
    script = tmp_path / "dtensor_overlap.py"
    script.write_text(_SCRIPT)
    out = subprocess.run(
        [sys.executable, str(script), _PATH, name], capture_output = True, text = True, timeout = 600
    )
    assert out.returncode == 0, out.stderr[-3000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("OVERLAPPED")][-1]
    overlapped = int(line.split()[1])
    # Four steps after the first (no state yet) overlap on CUDA; ROCm stays serial.
    assert overlapped == (0 if getattr(torch.version, "hip", None) else 4), line
