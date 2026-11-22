# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_oracle import LLAMA, build, tpn  # noqa: E402
import test_tp_native_gloo as gloo  # noqa: E402


def _case_tp_fsdp2_writer(
    rank,
    world,
    tmp,
    flattened = False,
    layout = None,
    earlier_model = False,
):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from test_fsdp2_native_units import _load_native

    native = _load_native()
    if not native._supports_ignored_params(native._fsdp_api()[2]):
        return {"unsupported": True}
    from unsloth.models import fsdp2_save

    mesh = init_device_mesh(
        "cpu",
        (2, 1, world // 2) if flattened else (2, world // 2),
        mesh_dim_names = ("dp", "cp", "tp") if flattened else ("dp", "tp"),
    )
    dp = mesh["dp", "cp"]._flatten("dp_cp") if flattened else mesh["dp"]
    if earlier_model:
        # An earlier tp x FSDP2 model on an equal mesh in the same process: DTensor ops can hand the
        # next model's tp parameters that earlier (equal, not identical) mesh.
        earlier = init_device_mesh("cpu", (2, world // 2), mesh_dim_names = ("dp", "tp"))
        first = build(LLAMA)
        tpn.apply_tensor_parallel(first, earlier["tp"])
        native.shard_model(first, native.FSDP2Config(param_dtype = torch.float32, mesh = earlier["dp"]))
    model = build(LLAMA)
    tpn.apply_tensor_parallel(model, mesh["tp"])
    config = dict(param_dtype = torch.float32, mesh = dp)
    if layout == "unsharded":
        config.update(reshard_after_forward = False, reshard_last_block = False)
    elif layout in ("ignored", "ignored_compiled"):
        # Every trainable tp parameter left out of FSDP2 (#16180): only tp-mesh DTensors remain.
        config["ignored_params"] = {p for p in model.parameters() if p.requires_grad}
    native.shard_model(model, native.FSDP2Config(**config))
    if layout == "unsharded":
        # An eval forward leaves every unit gathered: the parameters are tp-only DTensors.
        with torch.no_grad():
            model(input_ids = torch.zeros(1, 4, dtype = torch.long), use_cache = False)
    if layout == "ignored_compiled":
        # The recorded dp mesh sits on the root, not on the OptimizedModule wrapper.
        model = torch.compile(model, backend = "eager")
    # Before any gather: native_full_state_dict and native_save_optimizer read the writer as is.
    writer_rank = fsdp2_save.mesh_writer_rank(model)
    state = native.native_full_state_dict(model, trainable_only = True)
    with fsdp2_save.fsdp2_gathered(model, trainable_only = True) as writer:
        gathered = bool(writer)
    dist.barrier()
    return {
        "module": fsdp2_save.__file__,
        "writer_rank": writer_rank,
        "writer_rank_after": fsdp2_save.mesh_writer_rank(model),
        "gathered": gathered,
        "state_keys": len(state),
    }


gloo.CASES["tp_fsdp2_writer"] = _case_tp_fsdp2_writer


def _spawn(
    world,
    flattened,
    layout = None,
    earlier_model = False,
):
    import tempfile

    ctx = multiprocessing.get_context("spawn")
    port = gloo._free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(
                target = _worker,
                args = (
                    r,
                    world,
                    port,
                    "tp_fsdp2_writer",
                    tmp,
                    {"flattened": flattened, "layout": layout, "earlier_model": earlier_model},
                ),
            )
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


@pytest.mark.parametrize(
    "flattened, layout, earlier_model",
    [
        (False, None, False),
        (True, None, False),
        (False, "unsharded", False),
        (False, "ignored", False),
        (False, "ignored_compiled", False),
        (False, "unsharded", True),
        (False, "ignored", True),
    ],
    ids = [
        "sharded",
        "flattened",
        "unsharded_eval",
        "all_trainable_ignored",
        "all_trainable_ignored_compiled",
        "unsharded_eval_after_earlier_model",
        "all_trainable_ignored_after_earlier_model",
    ],
)
def test_tp_fsdp2_model_has_one_writer(flattened, layout, earlier_model):
    results = _spawn(4, flattened, layout, earlier_model)
    if results[0].get("unsupported"):
        pytest.skip(reason = "torch < 2.7: no fully_shard(ignored_params)")
    print("fsdp2_save:", results[0]["module"])
    assert [r["writer_rank"] for r in results] == [0, 0, 0, 0], results
    assert [r["writer_rank_after"] for r in results] == [0, 0, 0, 0], results
    assert [r["gathered"] for r in results] == [True, False, False, False], results
    assert results[0]["state_keys"] > 0, results
    assert [r["state_keys"] for r in results[1:]] == [0, 0, 0], results
