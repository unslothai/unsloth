# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pipeline-parallel 4-bit quantize-on-load sizes transformers' allocator warmup to this rank's stages and loads
synchronously, as the FSDP2 and TP sharded loads do. transformers 5 reserves get_total_byte_count bytes on every GPU
before the load (caching_allocator_warmup), so the whole model on each rank OOMs exactly the model-bigger-than-one-GPU
case pipeline parallel exists for. The warmup is recorded on a gloo CPU world of 2 ranks (it skips CPU devices, so
the recorder asks get_total_byte_count as if each parameter went to cuda:0). A load outside the pipeline context is
unchanged: the whole model and transformers' own async setting."""

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

pytest.importorskip("bitsandbytes")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pp_load_gloo import _free_port, _pp_load, _save_checkpoint, ppn  # noqa: E402
from test_pp_load_quantize_on_load_gloo import _load4  # noqa: E402


def _record_warmup():
    """Replace caching_allocator_warmup with a recorder of the bytes it would reserve (and the async setting it
    runs under); returns (records, undo)."""
    from transformers import core_model_loading, modeling_utils as mu

    if not callable(getattr(core_model_loading, "set_param_for_module", None)):
        pytest.skip(
            "transformers without core_model_loading.set_param_for_module (4.x) loads in full"
        )
    if not callable(mu.__dict__.get("get_total_byte_count")):
        pytest.skip("transformers without get_total_byte_count")
    records = []
    original = mu.caching_allocator_warmup

    def recorder(model, expanded_device_map, hf_quantizer):
        cuda = {n: torch.device("cuda", 0) for n in expanded_device_map}
        counts = mu.get_total_byte_count(model, cuda, hf_quantizer)
        records.append(
            {
                "bytes": int(sum(counts.values())),
                "async_off": os.environ.get("HF_DEACTIVATE_ASYNC_LOAD"),
            }
        )

    mu.caching_allocator_warmup = recorder

    def undo():
        mu.caching_allocator_warmup = original

    return records, undo, mu


def _case(rank, world, ckpt):
    from torch.distributed.device_mesh import init_device_mesh

    ppl = _pp_load()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    records, undo, mu = _record_warmup()
    counter = mu.get_total_byte_count
    try:
        whole = _load4(ckpt)  # outside the pipeline context: the whole model
        config = ppn.PPConfig(layers_per_stage = [4, 4])
        with ppl.pipeline_sharded_loading(mesh, schedule = "gpipe", config = config) as state:
            model = _load4(ckpt)
    finally:
        undo()
    del whole
    return {
        "records": records,
        "loaded": len(state["models"]),
        "kept_bytes": model.__dict__["_unsloth_pp_preload"]["kept_bytes"],
        "counter_restored": mu.get_total_byte_count is counter,
        "async_after": os.environ.get("HF_DEACTIVATE_ASYNC_LOAD"),
    }


def _worker(rank, world, port, tmp, ckpt):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    os.environ.pop("HF_DEACTIVATE_ASYNC_LOAD", None)
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        try:
            res = _case(rank, world, ckpt)
        except BaseException as exc:  # reported, so a skip or refusal reads as a result, not a hang
            import traceback
            res = {
                "error": f"{type(exc).__name__}: {exc}"[:800],
                "tb": traceback.format_exc()[-3000:],
            }
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def test_pp_quantize_on_load_warms_only_this_ranks_stages():
    world = 2
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _save_checkpoint("llama", os.path.join(tmp, "ckpt"))
        ctx = multiprocessing.get_context("spawn")
        port = _free_port()
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, tmp, ckpt)) for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(240)
        assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
        res = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]
    for r in res:
        if "error" in r and "Skipped" in r["error"]:
            pytest.skip(r["error"])
        assert "error" not in r, r
    for r in res:
        assert r["loaded"] == 1 and len(r["records"]) == 2, r
        (plain, staged) = r["records"]
        # A load outside the pipeline context: the whole model, transformers' own async setting.
        assert plain["async_off"] is None, r
        # The pipeline load: synchronous, and only this rank's stages reserved.
        assert staged["async_off"] == "1", r
        assert 0 < staged["bytes"] < plain["bytes"], r
        assert r["counter_restored"] and r["async_after"] is None, r
    whole = res[0]["records"][0]["bytes"]
    assert res[1]["records"][0]["bytes"] == whole
    # Each rank reserves what it keeps: its layers, the embedding or head of its stage (the final norm on both).
    for r in res:
        assert r["records"][1]["bytes"] == r["kept_bytes"], r
    assert sum(r["records"][1]["bytes"] for r in res) < 1.01 * whole, [r["records"] for r in res]
