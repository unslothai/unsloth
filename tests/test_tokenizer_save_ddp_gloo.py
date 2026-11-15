# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DDP / FSDP2 ranks saving one tokenizer folder must not re-read each other's half-written config."""

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
from test_tp_native_oracle import LLAMA  # noqa: E402


def _case(rank, world, tmp):
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from unsloth.save import patch_saving_functions

    tok = AutoTokenizer.from_pretrained(LLAMA)
    patch_saving_functions(tok)
    original = tok.original_save_pretrained

    def interleaved(directory, *args, **kwargs):
        result = original(directory, *args, **kwargs)
        config = os.path.join(directory, "tokenizer_config.json")
        if rank == 1:
            # Another rank's save truncates the file while this one is still between its write and re-read.
            with open(config, "w"):
                time.sleep(3)
            result = original(directory, *args, **kwargs)
        else:
            time.sleep(1)
        return result

    tok.original_save_pretrained = interleaved
    shared = os.path.join(tmp, "shared")
    out = {"added_tokens": len(getattr(tok, "added_tokens_decoder", {}))}
    dist.barrier()
    out["err"] = _raised(lambda: tok.save_pretrained(shared))
    dist.barrier()
    import json

    with open(os.path.join(shared, "tokenizer_config.json")) as f:
        out["final_keys"] = sorted(json.load(f))
    out["leftovers"] = sorted(n for n in os.listdir(tmp) if ".unsloth_tp_rank" in n)
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


def test_ranks_saving_one_tokenizer_folder_do_not_read_each_others_partial_files():
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
    for rank, res in enumerate(results):
        assert res["added_tokens"] > 0, res  # the fix-up re-read only runs with added tokens
        assert res["err"] is None, (rank, res["err"])
        assert "added_tokens_decoder" in res["final_keys"], res
        assert res["leftovers"] == [], res
