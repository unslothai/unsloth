# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A sub-mesh writer (rank 2 of meshes [0, 1] / [2, 3]) saves the same bnb 4-bit model and 8-bit
optimizer state as rank 0, and gathers the optimizer export per parameter to its host."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import datetime  # noqa: E402
import importlib  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402

WORLD = 4


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import bitsandbytes as bnb

        def nf4(seed):
            torch.manual_seed(seed)
            layer = bnb.nn.Linear4bit(
                128,
                96,
                bias = False,
                compute_dtype = torch.bfloat16,
                quant_type = "nf4",
                quant_storage = torch.bfloat16,
            )
            layer.weight = bnb.nn.Params4bit(
                torch.randn(96, 128) * 0.05,
                requires_grad = False,
                quant_type = "nf4",
                quant_storage = torch.bfloat16,
                blocksize = 64,
                module = layer,
            )
            return layer.to("cpu")

        first, second = nf4(0), nf4(1)
        torch.manual_seed(5)
        up = torch.nn.Linear(96, 128, bias = False, dtype = torch.bfloat16).requires_grad_(False)
        self.layers = torch.nn.ModuleList([first, torch.nn.Sequential(up, second)])
        torch.manual_seed(7)
        self.head = torch.nn.Linear(96, 64)

    def forward(self, x):
        return self.head(self.layers[1](self.layers[0](x).to(torch.bfloat16)).float())


def _bytes(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).clone()


def _case(rank, outdir):
    import bitsandbytes as bnb
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp import fully_shard

    native = _load_native()
    save = importlib.import_module("unsloth.models.fsdp2_save")
    qlora = importlib.import_module("unsloth.models.fsdp2_qlora")
    bnb_optim = importlib.import_module("unsloth.models.fsdp2_bnb_optim")
    mesh = DeviceMesh("cpu", [[0, 1], [2, 3]], mesh_dim_names = ("pair", "shard"))["shard"]
    model = _Model()
    qlora.prepare_fsdp2_qlora(model)
    for layer in model.layers:
        fully_shard(layer, mesh = mesh)
    fully_shard(model, mesh = mesh)
    save.install_dcp_bnb_state_dict()
    res = {"writer": save.mesh_writer_rank(model)}
    state = native.native_full_state_dict(model)
    res["model"] = {k: (str(v.dtype), tuple(v.shape), _bytes(v)) for k, v in state.items()}

    optimizer = bnb.optim.AdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr = 1e-2, min_8bit_size = 16
    )
    res["localized"] = bnb_optim.localize_bnb_optimizer(optimizer)
    torch.manual_seed(3 + rank % 2)
    for _ in range(2):
        model(torch.randn(4, 128, dtype = torch.bfloat16)).pow(2).mean().backward()
        optimizer.step()
        optimizer.zero_grad()
    exported = []
    original = type(optimizer).state_dict

    def _spy(self):
        out = original(self)
        exported.extend(
            "none" if v is None else ("dtensor" if type(v) is not torch.Tensor else str(v.device))
            for entry in out["state"].values()
            for k, v in entry.items()
            if k != "step"
        )
        return out

    type(optimizer).state_dict = _spy
    try:
        path = os.path.join(outdir, f"optimizer_pair{rank // 2}.pt")
        native.native_save_optimizer(model, optimizer, path)
    finally:
        type(optimizer).state_dict = original
    res["exported"] = sorted(set(exported))
    saved = torch.load(path, weights_only = False)
    res["optimizer"] = {
        f"{i}/{k}": (str(v.dtype), tuple(v.shape), _bytes(v)) if torch.is_tensor(v) else v
        for i, entry in saved["state"].items()
        for k, v in entry.items()
    }
    return res


def _worker(rank, world, port, outdir):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
        UNSLOTH_FSDP2_SAVE_TIMEOUT = "60",
    )
    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 150)
    )
    try:
        res = _case(rank, outdir)
    except BaseException as exc:  # a rank must report, not vanish
        import traceback
        res = {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-3000:]}"}
    torch.save(res, os.path.join(outdir, f"result_{rank}.pt"))
    try:
        dist.barrier()  # rank 0 hosts the store the other pair still waits on
    finally:
        dist.destroy_process_group()


def _run(timeout = 300):
    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as outdir:
        port = _free_port()
        procs = [ctx.Process(target = _worker, args = (r, WORLD, port, outdir)) for r in range(WORLD)]
        for p in procs:
            p.start()
        deadline = time.monotonic() + timeout
        while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
            time.sleep(0.2)
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(30)
        results = {}
        for r in range(WORLD):
            path = os.path.join(outdir, f"result_{r}.pt")
            results[r] = torch.load(path, weights_only = False) if os.path.exists(path) else None
    for rank, r in results.items():
        assert r is not None, f"rank {rank} did not report (hung collective?)"
        assert "crash" not in r, r["crash"]
    return results


def _same(a, b):
    assert sorted(a) == sorted(b), sorted(set(a) ^ set(b))
    for key in a:
        if isinstance(a[key], tuple):
            assert a[key][:2] == b[key][:2] and torch.equal(a[key][2], b[key][2]), key
        else:
            assert a[key] == b[key], key


def test_sub_mesh_writer_saves_bnb_state_like_rank0_and_exports_per_parameter():
    pytest.importorskip("bitsandbytes")
    results = _run()
    assert [results[r]["writer"] for r in range(WORLD)] == [0, 0, 2, 2], results
    assert all(results[r]["localized"] for r in range(WORLD))
    assert results[0]["model"] and not results[1]["model"] and not results[3]["model"]
    assert any(k.endswith("weight.absmax") for k in results[2]["model"])
    _same(results[0]["model"], results[2]["model"])
    assert results[0]["optimizer"] and any(k.endswith("/state1") for k in results[0]["optimizer"])
    for rank in (1, 2, 3):
        _same(results[0]["optimizer"], results[rank]["optimizer"])
    assert results[0]["exported"] == ["cpu"] and results[1]["exported"] == ["none"], results
    assert results[2]["exported"] == ["cpu"], results[2]["exported"]
    assert results[3]["exported"] == ["none"], results[3]["exported"]
