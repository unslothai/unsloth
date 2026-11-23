# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bitsandbytes 4-bit quantize-on-load read per pipeline stage, on a gloo CPU world of 2 ranks: under
pipeline_sharded_loading and under UNSLOTH_PP_SIZE's auto loading (UNSLOTH_PP_SHARDED_LOAD=1), each rank keeps only
its stages' quantized layers (the embedding / head where their stage lives); what it keeps is bitwise the full
quantize-on-load's (packed bytes and QuantState), and LoRA training through the pipeline matches the full load split
by apply_pipeline_parallel."""

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
from test_pp_load_gloo import (  # noqa: E402
    _free_port,
    _pp_load,
    _save_checkpoint,
    _storage_bytes,
    _torch_pipelining_error,
    _train,
    batch,
    ppn,
)
from test_pp_load_gloo import _peft as _peft_all  # noqa: E402


def _bnb():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_use_double_quant = True,
        bnb_4bit_compute_dtype = torch.float32,
    )


def _load4(ckpt):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        ckpt,
        dtype = torch.float32,
        attn_implementation = "eager",
        device_map = {"": "cpu"},
        quantization_config = _bnb(),
    )


def _four_bit(model):
    return [p for p in model.parameters() if type(p).__name__ == "Params4bit"]


def _case(
    rank,
    world,
    tmp,
    ckpt,
    mode,
    schedule = "gpipe",
):
    from torch.distributed.device_mesh import init_device_mesh

    ppl = _pp_load()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    full = _load4(ckpt)
    full_bytes = _storage_bytes(full)
    full4 = sum(p.numel() for p in _four_bit(full))
    full = _peft_all(full)
    ids, labels = batch(full)
    # The tiny vocabulary-sized head outweighs the layers: an even layer split, so each rank holds half the layers.
    config = ppn.PPConfig(layers_per_stage = [4, 4])
    pp_full = ppn.apply_pipeline_parallel(
        full, mesh, n_microbatches = 2, schedule = schedule, config = config
    )
    if mode == "explicit":
        cm = ppl.pipeline_sharded_loading(mesh, schedule = schedule, config = config)
    else:
        os.environ["UNSLOTH_PP_SHARDED_LOAD"] = "1"
        cm = ppl._pipeline_loading(lambda: mesh, schedule, config, auto = True)
    with cm as state:
        model = _load4(ckpt)
    stats = dict(model.__dict__.get("_unsloth_pp_preload") or {})
    loaded_bytes = _storage_bytes(model)
    kept4 = sum(p.numel() for p in _four_bit(model) if not ppl.is_placeholder(p))
    placeholders = [p for p in model.parameters() if ppl.is_placeholder(p)]
    zero_stride = all(all(s == 0 for s in p.stride()) for p in placeholders)
    model = _peft_all(model)
    pp = ppn.apply_pipeline_parallel(
        model, mesh, n_microbatches = 2, schedule = schedule, config = config
    )
    a, b = full.state_dict(), model.state_dict()
    diffs = sorted(set(a) ^ set(b)) + [
        k
        for k in a
        if k in b
        and (
            not isinstance(a[k], torch.Tensor)
            or a[k].shape != b[k].shape
            or not torch.equal(a[k], b[k])
        )
    ]
    owned_4bit = sum(1 for k in b if k.endswith(".weight.quant_state.bitsandbytes__nf4"))
    return {
        "stats": stats,
        "models_loaded": len(state["models"]),
        "full_bytes": full_bytes,
        "loaded_bytes": loaded_bytes,
        "full4": full4,
        "kept4": kept4,
        "placeholders": len(placeholders),
        "zero_stride": zero_stride,
        "diffs": diffs,
        "owned_4bit": owned_4bit,
        "full": _train(full, pp_full, ids, labels)[0],
        "sharded": _train(model, pp, ids, labels)[0],
        "bounds": pp.bounds,
    }


def _worker(rank, world, port, tmp, kwargs):
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
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 180)
    )
    try:
        try:
            res = _case(rank, world, tmp, **kwargs)
        except (
            Exception
        ) as exc:  # reported, so a refusal on an old tree reads as a failure, not a hang
            import traceback
            res = {
                "error": f"{type(exc).__name__}: {exc}"[:800],
                "tb": traceback.format_exc()[-3000:],
            }
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope = "module")
def checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        yield _save_checkpoint("llama", os.path.join(tmp, "ckpt"))


@pytest.mark.parametrize("mode,schedule", [("explicit", "gpipe"), ("auto", "1f1b")])
def test_quantize_on_load_reads_per_stage(checkpoint, mode, schedule):
    if _torch_pipelining_error():
        pytest.skip(_torch_pipelining_error())
    world = 2
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    kwargs = dict(ckpt = checkpoint, mode = mode, schedule = schedule)
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, tmp, kwargs)) for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        assert [p.exitcode for p in procs] == [0] * world
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]
    for res in results:
        assert "error" not in res, res
    kept = 0
    for rank, res in enumerate(results):
        assert res["models_loaded"] == 1
        assert res["stats"].get("quantized_on_load") is True, res["stats"]
        assert res["placeholders"] > 0 and res["zero_stride"]
        # Each rank keeps only its stages' 4-bit weights, about half of the full quantize-on-load.
        assert 0 < res["kept4"] < 0.75 * res["full4"], (res["kept4"], res["full4"])
        assert res["loaded_bytes"] < 0.8 * res["full_bytes"], (
            res["loaded_bytes"],
            res["full_bytes"],
        )
        # Owned packed bytes, QuantStates, norms, embedding / head: bitwise the full load's.
        assert res["diffs"] == [], res["diffs"][:10]
        assert res["owned_4bit"] > 0
        assert res["sharded"] == pytest.approx(res["full"], rel = 1e-5), (res["sharded"], res["full"])
        kept += res["kept4"]
    assert kept == results[0]["full4"]  # the two ranks hold every 4-bit weight exactly once
