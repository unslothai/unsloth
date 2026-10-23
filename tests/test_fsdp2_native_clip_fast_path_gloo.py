# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""native_clip_grad_norm_ stacks the per-tensor norms once (one cast) when every local shard is
non-empty and the norms share a device and dtype. On a gloo CPU world of 2, for norm types 0, 1, 2 and
inf, the total norm and the clipped gradients are bitwise equal to the per-tensor computation it
replaces, both on that fast path and on the fallback (an empty local shard, mixed gradient dtypes)."""

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
    """Many small sharded Linears (the LoRA shape of the problem), optionally a 1-row one (empty
    shard on rank 1) and a float16 one (a second gradient dtype)."""

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
    """The per-tensor computation the fast path replaces (norm, cast to float64, stack)."""
    import torch.distributed as dist

    grads_sharded, grads_plain = [], []
    for q in params:
        g = q.grad
        if isinstance(g, native_dtensor):
            grads_sharded.append(g.to_local())
        else:
            grads_plain.append(g)

    def per_tensor(grads):
        if not grads:
            return torch.zeros(0, dtype = torch.float64)
        empty = math.inf if p < 0 else 0.0
        live = [g.detach() for g in grads if g.numel()]
        norms = iter(list(torch._foreach_norm(live, p)) if live else [])
        return torch.stack(
            [
                next(norms).double() if g.numel() else torch.tensor(empty, dtype = torch.float64)
                for g in grads
            ]
        )

    rep = per_tensor(grads_plain)
    sh = per_tensor(grads_sharded)
    if p == 0:
        counts = sh.clone()
        dist.all_reduce(counts, op = dist.ReduceOp.SUM)
        total = (rep > 0).sum().double() + (counts > 0).sum().double()
    elif p == math.inf:
        part = sh.max()
        dist.all_reduce(part, op = dist.ReduceOp.MAX)
        total = (
            torch.max(torch.stack([rep.max(), part]))
            if rep.numel()
            else torch.max(torch.stack([part]))
        )
    else:
        powers = sh.pow(p).sum()
        dist.all_reduce(powers, op = dist.ReduceOp.SUM)
        total = (rep.pow(p).sum() + powers).pow(1.0 / p)
    coef = torch.clamp(float(max_norm) / (total + 1e-6), max = 1.0).float()
    return total.float(), coef


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
            torch.equal(
                q.grad.to_local() if isinstance(q.grad, DTensor) else q.grad,
                w.clone().mul_(coef) if w.numel() else w,
            )
            for q, w in zip(params, want_grads)
        )
        res[str(norm_type)] = {
            "got": float(got),
            "want": float(want_total),
            "bitwise": bool(torch.equal(got, want_total)),
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
def test_clip_fast_path_bitwise_equal_to_per_tensor_norms(case):
    results = _run(case)
    for rank, r in results.items():
        for norm_type in NORM_TYPES:
            v = r[str(norm_type)]
            assert v["bitwise"], (case, rank, norm_type, v)
            assert v["grads_equal"], (case, rank, norm_type, v)
            if case == "empty":
                assert v["empty_local"] == (rank == 1), (rank, v)
            if case == "mixed":
                assert v["dtypes"] == ["torch.float16", "torch.float32"], v
