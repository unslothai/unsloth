# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""pipeline_sharded_loading of a VLM (gloo pp 2) defers text_only to the split; without it the split refuses."""

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
import _tiny_mm as tiny  # noqa: E402
from test_pp_load_gloo import _pp_load  # noqa: E402
from test_pp_native_gloo import REPO, _torch_pipelining_error, ppn  # noqa: E402
from test_pp_vlm_text_only_gloo import check_parity, pipeline, reference  # noqa: E402

pytest.importorskip("transformers.models.gemma4")

LAYERS = 4


def _load(ckpt):
    from transformers import AutoModelForImageTextToText
    return AutoModelForImageTextToText.from_pretrained(
        ckpt, dtype = torch.float32, attn_implementation = "eager"
    )


def _peft(model):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    config = LoraConfig(
        r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False, lora_dropout = 0.0
    )
    return get_peft_model(model, config).train()


def _case(rank, world, tmp, ckpt):
    from torch.distributed.device_mesh import init_device_mesh

    ppl = _pp_load()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    ids, labels = tiny.batch()
    out = {"repo": os.path.abspath(ppl.__file__)}
    ref = _peft(_load(ckpt))
    out["ref"] = reference(ref, ids, labels)
    text_only = ppn.PPConfig(text_only = True)
    try:
        with ppl.pipeline_sharded_loading(mesh, schedule = "gpipe", config = text_only):
            model = _load(ckpt)
    except ValueError as exc:
        out["load_error"] = str(exc)
        return out
    out["load_error"] = None
    out["placeholders"] = sum(ppl.is_placeholder(p) for p in model.parameters())
    model = _peft(model)
    pp = ppn.apply_pipeline_parallel(model, mesh, n_microbatches = 2, config = text_only)
    out["media_error"] = None
    try:
        pp.step(ids, labels, pixel_values = torch.zeros(1, 3, 8, 8))
    except ValueError as exc:
        out["media_error"] = str(exc)
    out["pp"] = pipeline(model, pp, ids, labels)
    with ppl.pipeline_sharded_loading(mesh, schedule = "gpipe"):
        other = _peft(_load(ckpt))
    try:
        ppn.apply_pipeline_parallel(other, mesh, n_microbatches = 2)
        out["no_flag"] = None
    except ValueError as exc:
        out["no_flag"] = str(exc)
    return out


def _worker(rank, world, port, tmp, ckpt):
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

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    finished = False
    try:
        torch.save(_case(rank, world, tmp, ckpt), os.path.join(tmp, f"result_{rank}.pt"))
        finished = True
    finally:
        dist.destroy_process_group()
    if finished:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _spawn(ckpt, world = 2):
    from tests._rendezvous_port import free_port

    ctx = multiprocessing.get_context("spawn")
    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, tmp, ckpt)) for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(300)
        codes = [p.exitcode for p in procs]
        for p in procs:
            if p.is_alive():
                p.kill()
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


@pytest.mark.parametrize("arch", ["gemma4_unified", "gemma4"])
def test_sharded_load_of_multimodal_checkpoint_trains_text_only(arch):
    if _torch_pipelining_error():
        pytest.skip(_torch_pipelining_error())
    with tempfile.TemporaryDirectory() as ckpt:
        model = tiny.build(arch, LAYERS, lora = False)
        model.generation_config.pad_token_id = None
        model.save_pretrained(ckpt, max_shard_size = "1MB", safe_serialization = True)
        results = _spawn(ckpt)
    for res in results:
        assert res["repo"].startswith(REPO), (res["repo"], REPO)
        assert res["load_error"] is None, res["load_error"]
        assert res["placeholders"] > 0
        assert "not dropped" in (res["media_error"] or ""), res["media_error"]
        assert "PPConfig(text_only = True)" in (res["no_flag"] or ""), res["no_flag"]
    print(arch, "worst gradient error", check_parity(results))
