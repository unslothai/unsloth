# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""native_clip_grad_norm_ matches torch.nn.utils.clip_grad_norm_ in dtype and value on a 2-rank gloo
world; a float16 sharded norm above 256 stays finite."""

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

NORM_TYPES = (0.0, 1.0, 2.0, 3.0, math.inf, -math.inf)
# 16-bit: shard norm + combine round once more than torch's single norm
CASES = {
    "float32": (torch.float32, 1e-5),
    "bfloat16": (torch.bfloat16, 2**-6),
    "float16": (torch.float16, 2**-8),
    "mixed": (None, 1e-5),
    # complex64 norms are float32: the rank with the empty shard must gather float32 too
    "complex64": (torch.complex64, 1e-5),
}
SHAPES = ((3, 4), (1, 5), (6, 2), (4,))  # (3, 4) uneven over 2 ranks, (1, 5) empty on rank 1


def _grads(dtype, scale):
    g = torch.Generator().manual_seed(7)
    out = []
    for i, shape in enumerate(SHAPES):
        d = dtype or (torch.bfloat16 if i % 2 else torch.float32)
        t = torch.randn(*shape, generator = g) * scale
        if d.is_complex:
            t = torch.complex(t, torch.randn(*shape, generator = g) * scale)
        if i == 2:
            t[0] = 0.0  # a zero row: p = 0 counts tensors, -inf sees a zero
        out.append(t.to(d))
    return out


def _case(native, rank):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard, distribute_tensor

    mesh = init_device_mesh("cpu", (2,))
    res = {}
    for name, (dtype, _) in CASES.items():
        for scale in (1.0, 200.0) if name == "float16" else (1.0,):
            for norm_type in NORM_TYPES:
                full = _grads(dtype, scale)
                ref = [torch.nn.Parameter(torch.zeros_like(t)) for t in full]
                for q, t in zip(ref, full):
                    q.grad = t.clone()
                params = []
                for i, t in enumerate(full):
                    if i == len(full) - 1:  # a replicated (ignored_params) gradient
                        q = torch.nn.Parameter(torch.zeros_like(t))
                        q.grad = t.clone()
                    else:
                        q = torch.nn.Parameter(
                            distribute_tensor(torch.zeros_like(t), mesh, [Shard(0)])
                        )
                        q.grad = distribute_tensor(t.clone(), mesh, [Shard(0)])
                    params.append(q)
                want = torch.nn.utils.clip_grad_norm_(ref, 0.5, norm_type = norm_type)
                got = native.native_clip_grad_norm_(params, 0.5, norm_type = norm_type)
                diff = 0.0
                for q, r in zip(params, ref):
                    gq = q.grad.full_tensor() if isinstance(q.grad, DTensor) else q.grad
                    scale_r = max(r.grad.abs().max().item(), 1e-30)
                    diff = max(diff, ((gq - r.grad).abs().float().max() / scale_r).item())
                res[f"{name}|{scale}|{norm_type}"] = {
                    "got": float(got),
                    "want": float(want),
                    "got_dtype": str(got.dtype),
                    "want_dtype": str(want.dtype),
                    "grad_rel_diff": diff,
                }
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
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 30)
    )
    try:
        out.put((rank, _case(_load_native(), rank)))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(timeout = 180):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = mp.start_processes(
        _worker, args = (2, _free_port(), out), nprocs = 2, join = False, start_method = "spawn"
    )
    deadline = time.time() + timeout
    try:
        while not procs.join(timeout = 5):
            if time.time() > deadline:
                for p in procs.processes:
                    p.kill()
                pytest.fail(f"the ranks did not finish within {timeout} s (hung collective)")
    except mp.ProcessExitedException as exc:
        pytest.fail(f"a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for r in results.values():
        assert "crash" not in r, r["crash"]
    return results


@pytest.fixture(scope = "module")
def results():
    return _run()


@pytest.mark.parametrize("case", list(CASES))
def test_native_clip_follows_torch_dtype_and_value(results, case):
    tol = CASES[case][1]
    for rank, r in results.items():
        keys = [k for k in r if k.startswith(case + "|")]
        assert keys, (rank, case)
        for k in keys:
            v = r[k]
            assert v["got_dtype"] == v["want_dtype"], (rank, k, v)
            assert math.isfinite(v["got"]) == math.isfinite(v["want"]), (rank, k, v)
            if math.isfinite(v["want"]):
                assert math.isclose(v["got"], v["want"], rel_tol = tol, abs_tol = 1e-6), (rank, k, v)
            assert v["grad_rel_diff"] <= 2 * tol, (rank, k, v)
    assert all(results[0][k] == results[1][k] for k in results[0] if k.startswith(case + "|"))


def test_float16_sharded_norm_above_256_stays_finite(results):
    for rank, r in results.items():
        v = r["float16|200.0|2.0"]
        assert v["want"] > 256 and math.isfinite(v["want"]), v
        assert math.isfinite(v["got"]), (rank, v)
