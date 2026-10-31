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
    tp_native.apply_tensor_parallel(
        model, init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))["tp"]
    )
    writers = []
    original = tok.original_save_pretrained

    def recording(*args, **kwargs):
        if dist.get_rank() == 0:
            time.sleep(2)  # a rank that returns before rank 0 wrote sees no files
        writers.append(dist.get_rank())
        return original(*args, **kwargs)

    tok.original_save_pretrained = recording
    out = {"marked": bool(tok.__dict__.get("_unsloth_tp_world_save"))}

    def config(directory):
        import json
        with open(os.path.join(directory, "tokenizer_config.json")) as f:
            return json.load(f).get("model_max_length")

    def leftovers():
        return sorted(n for n in os.listdir(tmp) if ".unsloth_tp_rank" in n)

    # 1. one shared directory, every rank at once: complete files on every rank when it returns
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
    # 3. rank 0 saves alone, then every rank saves new state to the SAME directory
    same = os.path.join(tmp, "same")
    tok.model_max_length = 111
    if rank == 0:
        out["alone_err"] = _raised(lambda: tok.save_pretrained(same))
    dist.barrier()
    tok.model_max_length = 222
    out["after_return"] = tok.save_pretrained(same)
    out["after_on_return"] = config(same)
    dist.barrier()
    out["after_final"] = config(same)
    out["leftovers"] = leftovers()
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
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    return results


def test_tp2_kept_tokenizer_saves_complete_files_on_every_rank():
    for rank, res in enumerate(_spawn(2)):
        assert res["marked"], rank
        for key in ("shared_err", "own_err", "after_err"):
            assert res.get(key) is None, (rank, key, res.get(key))
        assert res["after_on_return"] == 222 and res["after_final"] == 222, (rank, res)
        assert res["shared_on_return"], (rank, res)
        assert res["own_on_return"], (rank, res)  # its own directory, written
        assert res["shared_writers"] == [
            rank
        ], res  # its own private copy, moved in with os.replace
        paths = [p for p in res["after_return"] if str(p).endswith("tokenizer_config.json")]
        assert paths and os.path.basename(os.path.dirname(paths[0])) == "same", res["after_return"]
        assert res["leftovers"] == [], res["leftovers"]  # private directories removed


class _PicklableProcessor:
    """A processor stand-in with its own save_pretrained (mark_world_save wraps it)."""

    def __init__(self):
        self.saved = []

    def save_pretrained(self, save_directory, **kwargs):
        self.saved.append(save_directory)
        return (save_directory,)


def test_marked_processor_stays_picklable(tmp_path):
    """Spawn-based DataLoader workers pickle the processor (and any collator holding it)."""
    import pickle

    from unsloth.models.tp_replica import mark_world_save

    processor = mark_world_save(_PicklableProcessor())
    assert processor.__dict__["save_pretrained"].__name__ == "unsloth_tp_world_save_pretrained"
    clone = pickle.loads(pickle.dumps(processor))
    # Unpickled in a worker, it is the plain class method again.
    assert clone.save_pretrained.__name__ == "save_pretrained"
    assert clone.save_pretrained(str(tmp_path)) == (str(tmp_path),)
    # Outside torch.distributed the wrapped save still writes through the object's own save.
    assert processor.save_pretrained(str(tmp_path / "out"))
    assert processor.saved
