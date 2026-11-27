# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor parallel Trainer checkpoints (gloo CPU, 2 ranks):

- a persistent for_inference replica in at save / resume time is released first (as pipeline parallel does), so
  the 1-GPU optimizer.pt holds every state and a resume restores the shards the optimizer steps;
- an optimizer state kept host-backed by UNSLOTH_OFFLOAD_OPTIMIZER_STATE is gathered from the parameter's
  device (NCCL gathers device tensors only; mocked here by a parameter on the meta device), and a resume on
  another layout puts the loaded state back on the host."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_gloo import _free_port  # noqa: E402
from test_tp_native_oracle import LLAMA, _batch, build, tpn  # noqa: E402


def _local(t):
    return t._local_tensor if tpn._is_dtensor(t) else t


def _trainer(model, optimizer, rank):
    return types.SimpleNamespace(
        model = model,
        optimizer = optimizer,
        lr_scheduler = None,
        args = types.SimpleNamespace(should_save = rank == 0),
    )


def _case_replica(rank, world, tmp):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from unsloth.models import tp_replica

    model = build(LLAMA).train()
    tpn.apply_tensor_parallel(model, init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",)))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-3)
    ids = _batch(model)
    model(input_ids = ids, labels = ids, use_cache = False).loss.backward()
    opt.step()
    trainer, out = _trainer(model, opt, rank), {}

    tp_replica.tp_for_inference(
        model
    )  # a callback's FastLanguageModel.for_inference before the save
    out["replica_in"] = tp_replica.replica_active(model)
    ckpt = os.path.join(tmp, "ckpt")
    tpn._tp_save_optimizer_and_scheduler(trainer, ckpt)
    dist.barrier()
    out["released_by_save"] = not tp_replica.replica_active(model)
    out["optimizer_states"] = len(opt.state)
    if rank == 0:
        portable = torch.load(os.path.join(ckpt, "optimizer.pt"), weights_only = False)
        out["portable_states"] = len(portable["state"])
    payload = torch.load(tpn._ckpt_path(ckpt, rank), weights_only = False)
    local = {n: tuple(_local(p).shape) for n, p in model.named_parameters() if p.requires_grad}
    out["trainables_local"] = all(
        tuple(v.shape) == local[n] for n, v in payload["trainables"].items()
    )
    out["trainables_count"] = (len(payload["trainables"]), len(local))

    tpn.save_tp_adapter(model, ckpt)
    saved = {n: _local(p).detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                _local(p).add_(1.0)  # weight drift the resume must undo
    tp_replica.tp_for_inference(
        model
    )  # train, for_inference, generate, then trainer.train(resume_from_checkpoint)
    tpn._tp_load_from_checkpoint(trainer, ckpt)
    out["released_by_load"] = not tp_replica.replica_active(model)
    out["resume_err"] = max(
        (_local(p) - saved[n]).abs().max().item()
        for n, p in model.named_parameters()
        if p.requires_grad
    )
    tp_replica.tp_for_inference(model)
    tpn._tp_load_optimizer_and_scheduler(trainer, ckpt)
    out["released_by_optimizer_load"] = not tp_replica.replica_active(model)
    return out


class _Offloader:
    def __init__(self):
        self.calls = []

    def offload(
        self,
        params = None,
        device_of = None,
    ):
        self.calls.append("offload")
        return set()

    def finish(self, devices):
        self.calls.append("finish")


def _case_offload(rank, world, tmp):
    from torch.distributed.device_mesh import init_device_mesh
    from unsloth.models import portable_resume

    DTensor, Shard = tpn._dtensor_api()[0], tpn._dtensor_api()[3]
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    out = {}
    # A parameter shard on another device than the state's host-backed shard: the gather must see the
    # parameter's device (on GPU that is the NCCL device; here the meta device stands in for it).
    shape, local_shape = (2 * world, 3), (2, 3)
    p = DTensor.from_local(
        torch.empty(local_shape, device = "meta"),
        mesh,
        [Shard(0)],
        run_check = False,
        shape = shape,
        stride = (3, 1),
    )
    host = DTensor.from_local(
        torch.randn(local_shape), mesh, [Shard(0)], run_check = False, shape = shape, stride = (3, 1)
    )
    seen = []
    real_full = tpn._full

    def recording_full(
        tensor,
        param = None,
        layout = None,
    ):
        seen.append(_local(tensor).device.type)
        return _local(tensor)

    tpn._full = recording_full
    try:
        portable_resume._full_value(torch.nn.Module(), p, host)
    finally:
        tpn._full = real_full
    out["gather_device"] = seen

    # A resume on another layout loads through the optimizer class method: the offloader must run after it.
    model = torch.nn.Linear(3, 2)
    inner = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    offloader = _Offloader()
    inner._unsloth_state_offloader = offloader
    state = {
        name: {
            "step": torch.tensor(3.0),
            "exp_avg": torch.full_like(t, 0.5),
            "exp_avg_sq": torch.full_like(t, 0.25),
        }
        for name, t in model.named_parameters()
    }
    group = {k: v for k, v in inner.param_groups[0].items() if k != "params"}
    named = {
        "state": state,
        "param_groups": [{**group, "params": [n for n, _ in model.named_parameters()]}],
    }
    portable_resume.portable_load_optimizer(model, inner, named)
    out["offloader_calls"] = offloader.calls
    out["loaded"] = all(
        torch.equal(inner.state[p]["exp_avg"], torch.full_like(p, 0.5)) for p in model.parameters()
    )
    return out


CASES = {"replica": _case_replica, "offload": _case_offload}


def _worker(rank, world, port, case, tmp):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        torch.save(CASES[case](rank, world, tmp), os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(case, world = 2):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp)) for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


def test_tp_checkpoint_releases_the_inference_replica():
    for rank, res in enumerate(_spawn("replica")):
        assert res["replica_in"], res
        assert res["released_by_save"] and res["released_by_load"], res
        assert res["released_by_optimizer_load"], res
        assert res["trainables_local"], res
        assert res["trainables_count"][0] == res["trainables_count"][1], res
        assert res["resume_err"] == 0.0, res
        if rank == 0:
            # Every optimizer state in the 1-GPU file (main wrote 0: the replica's tensors matched none).
            assert res["portable_states"] == res["optimizer_states"] > 0, res


def test_host_backed_optimizer_state_gathers_on_the_parameter_device():
    for res in _spawn("offload"):
        assert res["gather_device"] == ["meta"], res
        assert res["offloader_calls"] == ["offload", "finish"], res
        assert res["loaded"], res
