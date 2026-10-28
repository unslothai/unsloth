# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""native_fsdp2_unsharded after an eval forward with int reshard_after_forward used to clear the
post-forward mesh before unshard() ("Expected mesh_info to be FSDPMeshInfo, got NoneType")."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402

WORLD = 4


def _model():
    torch.manual_seed(42)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 8), torch.nn.Tanh(), torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)
    )


def _groups(native, model):
    return [g for unit in (model[0], model[2]) for g in native._unit_param_groups(unit)]


def _case(native, warm):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    reference = _model()
    x = torch.ones(2, 8)
    with torch.no_grad():
        want = reference(x)
    model = _model()
    mesh = init_device_mesh("cpu", (WORLD,))
    native.shard_model(
        model,
        native.FSDP2Config(
            mesh = mesh,
            device = "cpu",
            units = [model[0], model[2]],
            reshard_after_forward = 2,
            reshard_last_block = True,
            fit_layout = False,
            param_dtype = torch.float32,
            reduce_dtype = torch.float32,
        ),
    )
    res = {}
    if warm:
        with torch.no_grad():
            model(x)
        res["post_forward_before"] = any(g.is_sharded_post_forward for g in _groups(native, model))
    before = [g.post_forward_mesh_info for g in _groups(native, model)]
    try:
        with native.native_fsdp2_unsharded(model):
            res["gathered"] = all(not isinstance(model[i].weight, DTensor) for i in (0, 2))
            with torch.no_grad():
                res["max_diff"] = float((model(x) - want).abs().max())
            res["gathered_after_forward"] = all(
                not isinstance(model[i].weight, DTensor) for i in (0, 2)
            )
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        return res
    after = [g.post_forward_mesh_info for g in _groups(native, model)]
    res["restored"] = all(a is b for a, b in zip(after, before)) and all(
        b is not None for b in before
    )
    res["sharded_after"] = all(isinstance(model[i].weight, DTensor) for i in (0, 2))
    opt = torch.optim.AdamW(model.parameters(), lr = 0.01)
    loss = model(torch.randn(2, 8)).square().mean()
    loss.backward()
    opt.step()
    res["train_loss_finite"] = bool(torch.isfinite(loss.detach()))
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
        native = _load_native()
        out.put((rank, {warm: _case(native, warm) for warm in (False, True)}))
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

    out = mp.get_context("spawn").Queue()
    procs = mp.start_processes(
        _worker,
        args = (WORLD, _free_port(), out),
        nprocs = WORLD,
        join = False,
        start_method = "spawn",
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
    assert sorted(results) == list(range(WORLD)), results
    for rank, r in results.items():
        print("INT_MESH_UNSHARDED", rank, r)
        assert "crash" not in r, r["crash"]
    return results


@pytest.fixture(scope = "module")
def results():
    return _run()


@pytest.mark.parametrize("warm", [False, True], ids = ["cold", "after_eval_forward"])
def test_int_mesh_unsharded_context(results, warm):
    for rank, by_warm in results.items():
        r = by_warm[warm]
        assert "error" not in r, (rank, r)
        if warm:
            assert r["post_forward_before"], (rank, r)
        assert r["gathered"] and r["gathered_after_forward"], (rank, r)
        assert r["max_diff"] < 1e-6, (rank, r)
        assert r["restored"] and r["sharded_after"], (rank, r)
        assert r["train_loss_finite"], (rank, r)
