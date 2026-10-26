# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A tokenizer kept on a tensor parallel model saves once per world (rank 0 writes, the others wait), but only
when every rank saves to the same place: a rank saving to its own directory, or after a save rank 0 made alone
(`if rank == 0: tokenizer.save_pretrained(...)`), still ends with complete files of its own."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_gloo import _free_port, _import_checkout_unsloth, _raised  # noqa: E402
from test_tp_native_oracle import LLAMA, build  # noqa: E402


def _case(rank, world, tmp):
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoTokenizer
    from unsloth.models import tp_native
    from unsloth.save import patch_saving_functions

    tok = AutoTokenizer.from_pretrained(LLAMA)
    patch_saving_functions(tok)
    model = build(LLAMA)
    model.base_model.model.__dict__["_saved_temp_tokenizer"] = tok
    tp_native.apply_tensor_parallel(model, init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))["tp"])
    writers = []
    original = tok.original_save_pretrained

    def recording(*args, **kwargs):
        if dist.get_rank() == 0:
            time.sleep(2)  # a rank that returns before rank 0 wrote sees no files
        writers.append(dist.get_rank())
        return original(*args, **kwargs)

    tok.original_save_pretrained = recording
    out = {"marked": bool(tok.__dict__.get("_unsloth_tp_world_save"))}
    # 1. one shared directory: written once, by rank 0, before any rank returns
    shared = os.path.join(tmp, "shared")
    out["shared_err"] = _raised(lambda: tok.save_pretrained(shared))
    out["shared_on_return"] = os.path.isfile(os.path.join(shared, "tokenizer_config.json"))
    out["shared_writers"] = list(writers)
    dist.barrier()
    # 2. every rank its own directory: every directory written
    own = os.path.join(tmp, f"own_{rank}")
    out["own_err"] = _raised(lambda: tok.save_pretrained(own))
    out["own_on_return"] = os.path.isfile(os.path.join(own, "tokenizer_config.json"))
    dist.barrier()
    # 3. a save rank 0 makes alone, then one on every rank: files there when each rank returns
    if rank == 0:
        out["alone_err"] = _raised(lambda: tok.save_pretrained(os.path.join(tmp, "alone")))
    dist.barrier()
    after = os.path.join(tmp, "after")
    out["after_err"] = _raised(lambda: tok.save_pretrained(after))
    out["after_on_return"] = os.path.isfile(os.path.join(after, "tokenizer_config.json"))
    dist.barrier()
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


def _spawn(world = 2):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [ctx.Process(target = _worker, args = (r, world, port, tmp)) for r in range(world)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        assert [p.exitcode for p in procs] == [0] * world
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False) for r in range(world)
        ]
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    return results


def test_tp2_kept_tokenizer_saves_complete_files_on_every_rank():
    for rank, res in enumerate(_spawn(2)):
        assert res["marked"], rank
        for key in ("shared_err", "own_err", "after_err"):
            assert res[key] is None, (rank, key, res[key])
        assert res["shared_on_return"], (rank, res)
        assert res["own_on_return"], (rank, res)  # its own directory, written
        assert res["after_on_return"], (rank, res)  # not an earlier save rank 0 made alone
        if rank:
            assert res["shared_writers"] == [], res  # shared directory: rank 0 wrote it once
