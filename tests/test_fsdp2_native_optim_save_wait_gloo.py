# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""native_save_optimizer (gloo CPU world of 2): only rank 0 writes the full optimizer state, and the
other rank must not leave before that write finishes, or its next collective (the next step's
all-gather) waits on rank 0 for as long as a large torch.save takes and can hit the collective
timeout. Rank 0's write is slowed down; rank 1 must find the file complete when the call returns."""

import datetime
import os
import sys
import time

import pytest
import torch

if __import__("multiprocessing").parent_process() is not None or getattr(
    __import__("multiprocessing").current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_guards_gloo import _tiny_llama  # noqa: E402

DELAY_S = 3.0


def _worker(rank, world, init_file, path, out):
    os.environ.update(RANK = str(rank), WORLD_SIZE = str(world), LOCAL_RANK = str(rank))
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method = f"file://{init_file}",
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = 60),
    )
    try:
        native = _load_native()
        model = native.shard_model(_tiny_llama(), native.FSDP2Config(param_dtype = torch.float32))
        opt = torch.optim.AdamW(model.parameters(), lr = 1e-3, foreach = False)
        model(torch.tensor([[1, 2, 3]])).logits.float().mean().backward()
        opt.step()
        if rank == 0:
            save = torch.save

            def slow_save(*args, **kwargs):
                time.sleep(DELAY_S)  # a large state still being written
                return save(*args, **kwargs)

            torch.save = slow_save
        native.native_save_optimizer(model, opt, path, should_save = rank == 0)
        out.put((rank, {"file_complete_on_return": os.path.exists(path)}))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def test_non_writers_wait_for_the_optimizer_write(tmp_path):
    pytest.importorskip("transformers")
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    args = (2, str(tmp_path / "store"), str(tmp_path / "optimizer.pt"), out)
    procs = mp.start_processes(_worker, args = args, nprocs = 2, join = False, start_method = "spawn")
    deadline = time.time() + 180
    while not procs.join(timeout = 5):
        if time.time() > deadline:
            for p in procs.processes:
                p.kill()
            pytest.fail("the ranks did not finish within 180 s")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        assert "crash" not in r, r["crash"]
        assert r["file_complete_on_return"], (rank, results)
