# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""2 gloo ranks: a tensor parallel checkpoint whose optimizer was bitsandbytes AdamW8bit stepping the DTensor
shards locally resumes into torch AdamW (Studio steps sharded full finetunes with it), and the next step runs."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_gloo import _free_port  # noqa: E402
from test_tp_native_oracle import tpn  # noqa: E402


def _case(rank, world):
    try:
        import bitsandbytes as bnb
    except Exception as e:
        return {"skip": f"bitsandbytes is not importable: {e!r}"}
    from unsloth.models.fsdp2_bnb_optim import bnb_steps_cpu_tensors

    if not bnb_steps_cpu_tensors():
        return {"skip": "this bitsandbytes does not step CPU tensors"}
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    from unsloth.models.fsdp2_bnb_optim import _dequantize

    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    torch.manual_seed(0)
    full = torch.randn(8, 384)
    grads = [torch.randn(8, 384) for _ in range(4)]

    def param():
        return torch.nn.Parameter(distribute_tensor(full.clone(), mesh, [Shard(0)]))

    saved_param = param()
    saved_opt = bnb.optim.AdamW8bit([saved_param], lr = 1e-2, min_8bit_size = 0)
    tpn._tp_localize_bnb(saved_opt)
    for g in grads[:3]:
        saved_param.grad = distribute_tensor(g, mesh, [Shard(0)])
        saved_opt.step()
    saved = tpn._optimizer_local_state(saved_opt)
    entry = saved["state"][0]
    raw = entry.get("__bnb_optimizer_quant_state__", entry)
    local = saved_param._local_tensor
    moments = [
        _dequantize(raw, key, torch.device("cpu")).view(local.shape) for key in ("state1", "state2")
    ]

    resumed = param()
    opt = torch.optim.AdamW([resumed], lr = 1e-2)
    tpn._load_optimizer_local_state(opt, saved)
    state = opt.state[resumed]
    out = {
        "dtensor": type(state["exp_avg"]).__name__ == "DTensor",
        "exp_avg": torch.equal(state["exp_avg"].to_local(), moments[0]),
        "exp_avg_sq": torch.equal(state["exp_avg_sq"].to_local(), moments[1]),
    }
    resumed.grad = distribute_tensor(grads[3], mesh, [Shard(0)])
    before = resumed.to_local().clone()
    opt.step()
    out["stepped"] = not torch.equal(resumed.to_local(), before)
    return out


def _worker(rank, world, port, tmp):
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

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        try:
            res = _case(rank, world)
        except Exception as e:  # noqa: BLE001 -- reported to the parent
            import traceback
            res = {"error": f"{e!r}\n{traceback.format_exc()}"}
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_tp_bitsandbytes_checkpoint_resumes_into_torch_adamw():
    world = 2
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [ctx.Process(target = _worker, args = (r, world, port, tmp)) for r in range(world)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(300)
        assert [p.exitcode for p in procs] == [0] * world
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]
    if "skip" in results[0]:
        pytest.skip(reason = results[0]["skip"])
    for res in results:
        assert "error" not in res, res.get("error")
        assert res == {"dtensor": True, "exp_avg": True, "exp_avg_sq": True, "stepped": True}, res
