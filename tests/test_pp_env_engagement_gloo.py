# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_PP_SIZE with tp / dp inside the HF Trainer (test_pp_env_gloo's run): the other axis really engaged
(tp DTensor shards, FSDP2 shards, dp replicas on different rows) and the Trainer's registry query lands on the
registry cell that cites these runs. CPU gloo, 4 ranks."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_pp_env_gloo as base  # noqa: E402


def _engagement(model):
    from torch.distributed.fsdp import FSDPModule

    from unsloth.models import tp_native

    sharded = [
        n
        for n, p in model.named_parameters()
        if tp_native._is_dtensor(p) and any("Shard" in type(pl).__name__ for pl in p.placements)
    ]
    return {
        "is_tp": tp_native.is_tensor_parallel(model),
        "fsdp_modules": sum(isinstance(m, FSDPModule) for m in model.modules()),
        "sharded": sharded,
    }


def _worker(rank, world, port, tmp, env, ckpt):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    for key in base._ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(env)
    torch.set_num_threads(2)
    import torch.distributed as dist

    try:
        error = base._import_checkout_unsloth()
        if error is not None:
            torch.save({"skip": error}, os.path.join(tmp, f"result_{rank}.pt"))
            return
        import transformers

        from unsloth import parallel_capabilities as caps

        # The query the Trainer's own registry check builds (models/_utils.py check_layout).
        queries = []
        check = caps.check

        def recording_check(query, cells = True):
            if cells:
                status, cell = caps.cell_status(query)
                queries.append((query, status, getattr(cell, "name", cell)))
            return check(query, cells = cells)

        caps.check = recording_check
        base._import_checkout_unsloth = lambda: None  # keep this import (and the recorder)
        engaged = []
        train = transformers.Trainer.train

        def recording_train(self, *args, **kwargs):
            out = train(self, *args, **kwargs)
            if not engaged:
                engaged.append(_engagement(self.model))
            return out

        transformers.Trainer.train = recording_train
        res = base._case_env(rank, world, tmp, ckpt, 2, evaluate = False, post = False)
        res["queries"] = queries
        res["engaged"] = engaged[0]
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn(env, world = 4):
    ctx = multiprocessing.get_context("spawn")
    port = base._free_port()
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = base._checkpoint(tmp)
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, tmp, env, ckpt)) for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(int(os.environ.get("PP_TEST_TIMEOUT", 600)))
        codes = [p.exitcode for p in procs]
        for p in procs:
            if p.is_alive():
                p.kill()
        assert codes == [0] * world, f"rank exit codes {codes}"
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    return results


@base.needs_pipelining
@pytest.mark.parametrize(
    "env,cell",
    [
        ({"UNSLOTH_PP_SIZE": "2", "UNSLOTH_TP_SIZE": "2"}, "pp x tp"),
        ({"UNSLOTH_PP_SIZE": "2"}, "pp x dp (DDP)"),
        ({"UNSLOTH_PP_SIZE": "2", "UNSLOTH_FSDP2": "1"}, "pp x dp (native FSDP2)"),
    ],
    ids = ["pp2_x_tp2", "pp2_x_ddp2", "pp2_x_fsdp2_dp2"],
)
def test_pp_env_composed_axis_engages(env, cell):
    results = _spawn(env)
    tp = "UNSLOTH_TP_SIZE" in env
    fsdp2 = "UNSLOTH_FSDP2" in env
    for rank, res in enumerate(results):
        assert res["is_pp"] and res["queries"], (rank, res["queries"])
        for query, status, name in res["queries"]:
            assert status == "supported" and name.startswith(cell), (rank, status, name, query)
        engaged = res["engaged"]
        if res["stage"] == 1:  # the later stage ran on the first stage's rows
            assert res["stage_ids"] and sum(res["stage_ids"]) == sum(res["seen"]), res
        # tp: the stage's linear weights are tp DTensor shards; FSDP2: its modules are fully_shard roots.
        assert engaged["is_tp"] is tp, (rank, engaged)
        assert (engaged["fsdp_modules"] > 0) is fsdp2, (rank, engaged)
        assert bool(engaged["sharded"]) is (tp or fsdp2), (rank, engaged)
    if not tp:  # dp replicas trained on different rows; both stages of a replica on the same ones
        by_replica = {}
        for rank, res in enumerate(results):
            by_replica.setdefault(rank % 2, []).append(res["seen"])
        assert all(s == seen[0] for seen in by_replica.values() for s in seen), by_replica
        assert by_replica[0][0] != by_replica[1][0], by_replica
    else:  # tp peers of a stage saw the same rows
        by_stage = {}
        for res in results:
            by_stage.setdefault(res["stage"], []).append(res["seen"])
        assert all(len(v) == 2 and v[0] == v[1] for v in by_stage.values()), by_stage
