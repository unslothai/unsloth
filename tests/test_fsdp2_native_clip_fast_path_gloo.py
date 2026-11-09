# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Batched native_clip_grad_norm_ equals a one-tensor-at-a-time reference on a 2-rank gloo world
(bitwise, or within one float32 rounding), incl. empty shards and mixed dtypes."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402

NORM_TYPES = (0.0, 1.0, 2.0, math.inf)


class _Many(torch.nn.Module):
    """Many small sharded Linears, optionally a 1-row (empty shard) and a float16 one."""

    def __init__(
        self,
        empty_shard = False,
        mixed = False,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.layers = torch.nn.ModuleList(torch.nn.Linear(6, 6) for _ in range(12))
        self.one = torch.nn.Linear(6, 1) if empty_shard else None
        self.lin16 = torch.nn.Linear(6, 4).half() if mixed else None

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = torch.tanh(layer(h))
        out = h.square().sum()
        if self.one is not None:
            out = out + self.one(h).square().sum()
        if self.lin16 is not None:
            out = out + self.lin16(h.half()).float().square().sum()
        return out


def _reference(native_dtensor, params, max_norm, p):
    """The same computation one tensor at a time."""
    import torch.distributed as dist

    norms = {}  # (sharded, dtype) -> norms; order: plain then sharded
    for q in params:
        g = q.grad
        sharded = isinstance(g, native_dtensor)
        local = g.to_local() if sharded else g
        if local.numel():
            n = torch._foreach_norm([local.detach()], p)[0]
        else:
            n = torch.tensor(math.inf if p < 0 else 0.0, dtype = local.dtype)
        if sharded:
            parts = [torch.empty_like(n) for _ in range(dist.get_world_size())]
            dist.all_gather(parts, n)
            n = torch.linalg.vector_norm(torch.stack(parts), p)
        norms.setdefault((sharded, local.dtype), []).append(n)
    ordered = [n for key in sorted(norms, key = lambda k: k[0]) for n in norms[key]]
    total = torch.linalg.vector_norm(torch.stack(ordered), p)
    coef = torch.clamp(float(max_norm) / (total + 1e-6), max = 1.0)
    return total, coef


def _case(native, empty_shard, mixed):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor

    model = _Many(empty_shard = empty_shard, mixed = mixed)
    mesh = init_device_mesh("cpu", (2,))
    if model.lin16 is not None:
        fully_shard(model.lin16, mesh = mesh)
    fully_shard(model, mesh = mesh)
    x = torch.randn(3, 6, generator = torch.Generator().manual_seed(1))
    res = {}
    for norm_type in NORM_TYPES:
        model.zero_grad(set_to_none = True)
        model(x).backward()
        params = list(model.parameters())
        want_total, coef = _reference(DTensor, params, 0.05, norm_type)
        want_grads = [
            (q.grad.to_local() if isinstance(q.grad, DTensor) else q.grad).clone() for q in params
        ]
        got = native.native_clip_grad_norm_(params, 0.05, norm_type = norm_type)
        grads_equal = all(
            torch.allclose(
                q.grad.to_local() if isinstance(q.grad, DTensor) else q.grad,
                w.clone().mul_(coef) if w.numel() else w,
            )
            for q, w in zip(params, want_grads)
        )
        res[str(norm_type)] = {
            "got": float(got),
            "want": float(want_total),
            "bitwise": bool(torch.equal(got, want_total)),
            "close": bool(torch.allclose(got, want_total, rtol = 2**-22, atol = 0)),
            "grads_equal": grads_equal,
            "empty_local": any(
                isinstance(q.grad, DTensor) and q.grad.to_local().numel() == 0 for q in params
            ),
            "dtypes": sorted({str(q.grad.dtype) for q in params}),
        }
    return res


def _worker(rank, world, port, out, case):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        UNSLOTH_FSDP2 = "1",
    )
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 30)
    )
    try:
        native = _load_native()
        out.put((rank, _case(native, empty_shard = case == "empty", mixed = case == "mixed")))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(case, timeout = 180):
    import torch.multiprocessing as mp

    out = mp.get_context("spawn").Queue()
    procs = mp.start_processes(
        _worker, args = (2, _free_port(), out, case), nprocs = 2, join = False, start_method = "spawn"
    )
    deadline = time.time() + timeout
    try:
        while not procs.join(timeout = 5):
            if time.time() > deadline:
                for p in procs.processes:
                    p.kill()
                pytest.fail(f"{case}: the ranks did not finish within {timeout} s")
    except mp.ProcessExitedException as exc:
        pytest.fail(f"{case}: a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for r in results.values():
        assert "crash" not in r, r["crash"]
    return results


@pytest.mark.parametrize("case", ["fast", "empty", "mixed"])
def test_clip_batched_equal_to_per_tensor_norms(case):
    results = _run(case)
    for rank, r in results.items():
        for norm_type in NORM_TYPES:
            v = r[str(norm_type)]
            assert v["bitwise"] if case != "empty" else v["close"], (case, rank, norm_type, v)
            assert v["grads_equal"], (case, rank, norm_type, v)
            if case == "empty":
                assert v["empty_local"] == (rank == 1), (rank, v)
            if case == "mixed":
                assert v["dtypes"] == ["torch.float16", "torch.float32"], v
