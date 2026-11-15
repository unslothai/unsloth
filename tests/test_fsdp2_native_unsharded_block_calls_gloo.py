# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A decode loop calling blocks before any root forward made a block FSDP2's root; training must still run."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402

WORLD = 2


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8) for _ in range(2)])
        self.head = torch.nn.Linear(8, 1)

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.head(x)


def _loss(model, x):
    return model(x).square().mean()


def _case(native):
    from torch.distributed.device_mesh import init_device_mesh

    x = torch.arange(16, dtype = torch.float32).reshape(2, 8) / 16
    reference = _Net()
    want = _loss(reference, x)
    want.backward()
    want_grad = float(reference.head.weight.grad.norm())
    model = _Net()
    native.shard_model(
        model,
        native.FSDP2Config(
            mesh = init_device_mesh("cpu", (WORLD,)),
            device = "cpu",
            units = list(model.layers),
            fit_layout = False,
            param_dtype = torch.float32,
            reduce_dtype = torch.float32,
        ),
    )
    res = {}
    try:
        with native.native_fsdp2_unsharded(model), torch.no_grad():
            h = x
            for layer in model.layers:  # a decode loop: the blocks, never the root
                h = torch.tanh(layer(h))
        loss = _loss(model, x)
        loss.backward()
        res["loss_diff"] = abs(float(loss) - float(want))
        grad = model.head.weight.grad
        grad = grad.full_tensor() if hasattr(grad, "full_tensor") else grad
        res["grad_diff"] = abs(float(grad.norm()) - want_grad)
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return res


def _worker(rank, world, port, out):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 60)
    )
    try:
        out.put((rank, _case(_load_native())))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def test_training_forward_after_block_calls_in_the_gathered_context():
    import torch.multiprocessing as mp

    out = mp.get_context("spawn").Queue()
    procs = mp.start_processes(
        _worker, args = (WORLD, _free_port(), out), nprocs = WORLD, join = False, start_method = "spawn"
    )
    deadline = time.time() + 180
    while not procs.join(timeout = 5):
        if time.time() > deadline:
            for p in procs.processes:
                p.kill()
            pytest.fail("the ranks did not finish within 180 s (hung collective)")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == list(range(WORLD)), results
    for rank, r in results.items():
        assert "crash" not in r, r["crash"]
        assert "error" not in r, (rank, r["error"])
        assert r["loss_diff"] < 1e-6 and r["grad_diff"] < 1e-5, (rank, r)
