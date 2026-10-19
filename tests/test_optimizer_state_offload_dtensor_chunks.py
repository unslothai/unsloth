# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_OFFLOAD_OPTIMIZER_STATE chunked step under FSDP2: a DTensor's numel() is the global
one, but each rank holds (and moves to the device) only its local shard of the optimizer state.
Sizing chunks by the global numel overcounted by world_size, so the step ran in world_size times
more chunks than chunk_bytes allows. A fake 2-rank process group on CPU, no GPU."""

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_ROOT, "unsloth", "optimizers", "state_offload.py")

_SCRIPT = textwrap.dedent(
    """
    import importlib.util, sys, torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard
    from torch.testing._internal.distributed.fake_pg import FakeStore

    dist.init_process_group("fake", store = FakeStore(), rank = 0, world_size = 2)
    spec = importlib.util.spec_from_file_location("_unsloth_state_offload", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mesh = init_device_mesh("cpu", (2,))
    n = 1000  # global numel per parameter; this rank holds n // 2

    def shard(local):
        return DTensor.from_local(local, mesh, [Shard(0)], run_check = False)

    params = [torch.nn.Parameter(shard(torch.randn(n // 2))) for _ in range(2)]
    assert params[0].numel() == n and params[0]._local_tensor.numel() == n // 2
    optimizer = torch.optim.AdamW(params)
    # Never stepped: 2 fp32 moments of the local shard = 4n bytes per parameter on this rank.
    unstepped = module._StateOffloader(optimizer, chunk_bytes = 9 * n)._chunks(params)
    for p in params:
        optimizer.state[p] = {"exp_avg": shard(torch.zeros(n // 2)), "exp_avg_sq": shard(torch.zeros(n // 2))}
    stepped = module._StateOffloader(optimizer, chunk_bytes = 9 * n)._chunks(params)
    print("CHUNKS", len(unstepped), len(stepped))
    dist.destroy_process_group()
    """
)


def test_dtensor_state_is_sized_by_its_local_shard(tmp_path):
    pytest.importorskip("torch.distributed.tensor")
    try:
        from torch.testing._internal.distributed.fake_pg import FakeStore  # noqa: F401
    except Exception:
        pytest.skip("torch without the fake process group")
    script = tmp_path / "chunks.py"
    script.write_text(_SCRIPT)
    out = subprocess.run(
        [sys.executable, str(script), _PATH], capture_output = True, text = True, timeout = 300
    )
    assert out.returncode == 0, out.stderr[-3000:]
    line = [l for l in out.stdout.splitlines() if l.startswith("CHUNKS")][-1]
    # 2 parameters x 4n local bytes = 8n fit one 9n chunk; at the global numel each would be 8n.
    assert line.split()[1:] == ["1", "1"], line
