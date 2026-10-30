# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""dp 2 x tp 2 gloo: a tensor parallel x native FSDP2 LoRA model has one writer for a gathered save, not one
per tp row (its frozen tp DTensors come first, on the rank's own tp row)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_oracle import LLAMA, build, tpn  # noqa: E402
import test_tp_native_gloo as gloo  # noqa: E402


def _case_tp_fsdp2_writer(rank, world, tmp):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from test_fsdp2_native_units import _load_native

    native = _load_native()
    if not native._supports_ignored_params(native._fsdp_api()[2]):
        return {"unsupported": True}
    from unsloth.models import fsdp2_save

    mesh = init_device_mesh("cpu", (2, world // 2), mesh_dim_names = ("dp", "tp"))
    model = build(LLAMA)
    tpn.apply_tensor_parallel(model, mesh["tp"])
    native.shard_model(model, native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"]))
    with fsdp2_save.fsdp2_gathered(model, trainable_only = True) as writer:
        gathered = bool(writer)
    state = native.native_full_state_dict(model, trainable_only = True)
    dist.barrier()
    return {
        "module": fsdp2_save.__file__,
        "writer_rank": fsdp2_save.mesh_writer_rank(model),
        "gathered": gathered,
        "state_keys": len(state),
    }


gloo.CASES["tp_fsdp2_writer"] = _case_tp_fsdp2_writer


def _spawn(world):
    import tempfile

    ctx = multiprocessing.get_context("spawn")
    port = gloo._free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, "tp_fsdp2_writer", tmp, {}))
            for r in range(world)
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


def _worker(*args):
    gloo._worker(*args)


def test_tp_fsdp2_model_has_one_writer():
    results = _spawn(4)
    if results[0].get("unsupported"):
        pytest.skip("torch < 2.7: no fully_shard(ignored_params)")
    print("fsdp2_save:", results[0]["module"])
    assert [r["writer_rank"] for r in results] == [0, 0, 0, 0], results
    assert [r["gathered"] for r in results] == [True, False, False, False], results
    assert results[0]["state_keys"] > 0, results
    assert [r["state_keys"] for r in results[1:]] == [0, 0, 0], results
