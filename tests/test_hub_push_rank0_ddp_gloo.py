# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Plain DDP: Hub uploads run on rank 0 only; local saves run on every rank."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_gloo import _free_port, _import_checkout_unsloth  # noqa: E402


def _case(rank, world, tmp):
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    from unsloth.save import _tp_collective_save

    calls = []

    def push_to_hub(
        self,
        repo_id,
        token = None,
    ):
        calls.append(("push_to_hub", repo_id))
        return "pushed"

    def save_pretrained(
        self,
        save_directory,
        push_to_hub = False,
        token = None,
    ):
        calls.append(("save_pretrained", push_to_hub))
        return "saved"

    model = torch.nn.Linear(2, 2)
    push = _tp_collective_save("push_to_hub")(push_to_hub)
    save = _tp_collective_save("save_pretrained")(save_pretrained)
    out = {
        "push": push(model, "u/r", token = "x"),
        "save_push": save(model, "u/r", push_to_hub = True),
        "save_local": save(model, os.path.join(tmp, "local")),
    }
    out["calls"] = calls
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
        torch.save(_case(rank, world, tmp), os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_plain_ddp_uploads_from_rank0_only():
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [ctx.Process(target = _worker, args = (r, 2, port, tmp)) for r in range(2)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        assert [p.exitcode for p in procs] == [0, 0]
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False) for r in range(2)
        ]
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    rank0, rank1 = results
    assert rank0["calls"] == [
        ("push_to_hub", "u/r"),
        ("save_pretrained", True),
        ("save_pretrained", False),
    ]
    assert rank0["push"] == "pushed" and rank0["save_push"] == "saved"
    assert rank1["calls"] == [("save_pretrained", False)], rank1
    assert rank1["push"] is None and rank1["save_push"] is None


def test_fsdp1_models_keep_the_all_rank_save(monkeypatch):
    # FSDP1's state dict hooks all-gather on every rank: skipping the save on rank 1 would hang rank 0.
    import torch
    import torch.distributed as dist
    import torch.distributed.fsdp as fsdp
    from unsloth import save as unsloth_save

    class FakeFSDP1(torch.nn.Module):
        pass

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda *a: 2)
    monkeypatch.setattr(fsdp, "FullyShardedDataParallel", FakeFSDP1)
    plain = torch.nn.Sequential(torch.nn.Linear(2, 2))
    wrapped = torch.nn.Sequential(FakeFSDP1())
    push = lambda self, *a, **k: None
    assert unsloth_save._plain_data_parallel_push(plain, "push_to_hub", push, (), {})
    assert not unsloth_save._plain_data_parallel_push(wrapped, "push_to_hub", push, (), {})
