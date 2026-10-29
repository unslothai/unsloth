# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 keeps vLLM-shared frozen weights replicated (same storage) instead of sharding a copy (T4 OOM)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import datetime  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402
from test_fsdp2_native_trainer_args_gloo import _lora  # noqa: E402

CASES = ("no_engine", "engine_on_peft", "engine_on_inner", "engine_not_shared", "env_shards")


def _case(native, case):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor

    os.environ.pop("UNSLOTH_FSDP2_REPLICATE_FROZEN", None)
    model = _lora(1)
    if case == "engine_on_peft":
        model.vllm_engine = types.SimpleNamespace(shared_weights = True)
    elif case == "engine_on_inner":
        model.base_model.model.vllm_engine = types.SimpleNamespace(shared_weights = True)
    elif case == "engine_not_shared":
        model.vllm_engine = types.SimpleNamespace()
    elif case == "env_shards":
        model.vllm_engine = types.SimpleNamespace(shared_weights = True)
        os.environ["UNSLOTH_FSDP2_REPLICATE_FROZEN"] = "0"
    frozen = {n: p.data_ptr() for n, p in model.named_parameters() if not p.requires_grad}
    mesh = init_device_mesh("cpu", (2,))
    agreed = {}
    original = native._agree_over_ranks

    def record(*args, **kwargs):
        agreed["flags"] = original(*args, **kwargs)
        return agreed["flags"]

    native._agree_over_ranks = record
    try:
        cfg = native.FSDP2Config(param_dtype = torch.float32, device = "cpu", mesh = mesh)
        model = native.shard_model(model, cfg)
    finally:
        native._agree_over_ranks = original
    fits, replicate_frozen, _, _ = agreed["flags"]
    params = dict(model.named_parameters())
    return {
        "fits": fits,
        "replicate_frozen": replicate_frozen,
        "frozen_dtensor": sum(isinstance(params[n], DTensor) for n in frozen),
        "frozen_same_storage": sum(
            not isinstance(params[n], DTensor) and params[n].data_ptr() == ptr
            for n, ptr in frozen.items()
        ),
        "n_frozen": len(frozen),
        "trainable_dtensor": all(
            isinstance(p, DTensor) for p in params.values() if p.requires_grad
        ),
    }


def _worker(rank, world, port, out):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
    )
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 60)
    )
    try:
        native = _load_native()
        res = {}
        for case in CASES:
            try:
                res[case] = _case(native, case)
            except Exception as exc:
                import traceback
                res[case] = {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}
        out.put((rank, res))
    finally:
        dist.destroy_process_group()


def test_vllm_shared_frozen_weights_stay_replicated():
    pytest.importorskip("peft")
    import inspect

    from torch.distributed.fsdp import fully_shard

    if "ignored_params" not in inspect.signature(fully_shard).parameters:
        pytest.skip(
            "fully_shard has no ignored_params (torch < 2.7): frozen weights cannot stay replicated"
        )
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(
        _worker, args = (2, _free_port(), out), nprocs = 2, join = True, start_method = "spawn"
    )
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, res in results.items():
        for case, r in res.items():
            print("VLLM_SHARED_FROZEN", rank, case, r)
            assert "crash" not in r, (case, r["crash"])
            assert r["fits"] is False and r["trainable_dtensor"] and r["n_frozen"] > 0, (case, r)
            if case in ("engine_on_peft", "engine_on_inner"):
                assert r["replicate_frozen"] is True, (case, r)
                assert r["frozen_dtensor"] == 0 and r["frozen_same_storage"] == r["n_frozen"], (
                    case,
                    r,
                )
            else:
                assert r["replicate_frozen"] is False and r["frozen_dtensor"] == r["n_frozen"], (
                    case,
                    r,
                )
