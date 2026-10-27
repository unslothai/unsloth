# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""pipeline_sharded_loading on a gloo CPU world (2 / 4 ranks): each rank reads only its stages' layers (and the
embedding / head where their stage lives), and the split model trains bitwise like a full load split by
apply_pipeline_parallel; refusals before anything is cut."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib.util  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pp_native_gloo import (  # noqa: E402
    GEMMA3,
    LLAMA,
    QWEN3,
    REPO,
    _free_port,
    _torch_pipelining_error,
    batch,
    build,
    ppn,
)


def _pp_load():
    full = "unsloth.models.pp_load"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(
        full, os.path.join(REPO, "unsloth", "models", "pp_load.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


ARCHS = {"llama": LLAMA, "qwen3": QWEN3, "gemma3": GEMMA3}
N_LAYERS = 8


def _save_checkpoint(arch, directory, tie = None):
    model = build(ARCHS[arch], layers = N_LAYERS, lora = False, tie = tie)
    model.generation_config.pad_token_id = None  # tiny repos ship -1, which save_pretrained rejects
    model.save_pretrained(directory, max_shard_size = "1MB", safe_serialization = True)
    return directory


def _peft(model, **extra):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    kw = dict(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False, lora_dropout = 0.0)
    kw.update(extra)
    return get_peft_model(model, LoraConfig(**kw)).train()


def _load(ckpt):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype = torch.float32, attn_implementation = "eager")
    for key in ("attention_dropout", "hidden_dropout", "dropout"):
        if isinstance(getattr(model.config, key, None), float):
            setattr(model.config, key, 0.0)
    return model


def _storage_bytes(model):
    seen, total = set(), 0
    for p in model.parameters():
        storage = p.untyped_storage()
        if storage.data_ptr() in seen:
            continue
        seen.add(storage.data_ptr())
        total += storage.nbytes()
    return total


def _train(model, pp, ids, labels, steps = 3):
    opt = torch.optim.AdamW(pp.parameters(), lr = 1e-2, foreach = False)
    losses, grads = [], None
    for step in range(steps):
        losses.append(float(pp.step(ids, labels)))
        if step == 0:
            grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
        pp.clip_grad_norm_(1.0)
        opt.step()
        opt.zero_grad(set_to_none = True)
    return losses, grads


def _case_load(rank, world, tmp, ckpt, schedule = "gpipe", n_mb = 2, v = None):
    from torch.distributed.device_mesh import init_device_mesh

    ppl = _pp_load()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    config = ppn.PPConfig(stages_per_rank = v)
    full = _peft(_load(ckpt))
    full_lora = {n: p.detach().clone() for n, p in full.named_parameters() if "lora_" in n}
    full_bytes = _storage_bytes(full)
    ids, labels = batch(full)
    pp_full = ppn.apply_pipeline_parallel(full, mesh, n_microbatches = n_mb, schedule = schedule, config = config)
    full_after = _storage_bytes(full)
    seen, kept_numel = set(), 0
    for n, p in full.named_parameters():
        if "lora_" not in n and id(p) not in seen:
            seen.add(id(p))
            kept_numel += p.numel()
    with ppl.pipeline_sharded_loading(mesh, schedule = schedule, config = config) as state:
        model = _load(ckpt)
    stats = dict(model._unsloth_pp_preload)
    loaded_bytes = _storage_bytes(model)
    unsplit = None
    try:
        model(input_ids = ids)
    except RuntimeError as exc:
        unsplit = str(exc)
    model = _peft(model)
    lora_equal = set(full_lora) == {n for n, _ in model.named_parameters() if "lora_" in n} and all(
        torch.equal(full_lora[n], p) for n, p in model.named_parameters() if "lora_" in n
    )
    pp = ppn.apply_pipeline_parallel(model, mesh, n_microbatches = n_mb, schedule = schedule, config = config)
    sharded_after = _storage_bytes(model)
    # Same weights where this rank computes: every owned parameter / buffer bitwise equal to the full load.
    a, b = full.state_dict(), model.state_dict()
    diffs = sorted(set(a) ^ set(b)) + [
        k for k in a if k in b and not torch.equal(a[k], b[k])
    ]
    full_res = _train(full, pp_full, ids, labels)
    res = _train(model, pp, ids, labels)
    return {
        "stats": stats,
        "models_loaded": len(state["models"]),
        "bytes": {"full": full_bytes, "full_after": full_after, "loaded": loaded_bytes, "after": sharded_after},
        "unsplit": unsplit,
        "lora_equal": lora_equal,
        "diffs": diffs,
        "full": full_res,
        "sharded": res,
        "local": pp.local_stages,
        "bounds": pp.bounds,
        "kept_numel": kept_numel,
        "guards": len(model.base_model.model.__dict__.get("_unsloth_pp_preload_guards", ())),
    }


def _case_refusals(rank, world, tmp, ckpt):
    from torch.distributed.device_mesh import init_device_mesh

    ppl = _pp_load()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))["pp"]
    out = {}

    def catch(key, fn):
        try:
            fn()
            out[key] = None
        except (ValueError, RuntimeError) as exc:
            out[key] = str(exc)

    with ppl.pipeline_sharded_loading(mesh, schedule = "gpipe"):
        model = _peft(_load(ckpt))
    catch("other_split", lambda: ppn.apply_pipeline_parallel(model, mesh, n_microbatches = 4, schedule = "interleaved1f1b"))
    out["other_split:uncut"] = not ppn.is_pipeline_parallel(model)
    catch(
        "other_layers_per_stage",
        lambda: ppn.apply_pipeline_parallel(model, mesh, config = ppn.PPConfig(layers_per_stage = [1, N_LAYERS - 1])),
    )
    with ppl.pipeline_sharded_loading(mesh, schedule = "gpipe"):
        trainable_head = _peft(_load(ckpt), modules_to_save = ["lm_head"])
    catch("trainable_placeholder_head", lambda: ppn.apply_pipeline_parallel(trainable_head, mesh))
    out["trainable_placeholder_head:uncut"] = not ppn.is_pipeline_parallel(trainable_head)
    from transformers import AutoModelForCausalLM

    def quantized():
        with ppl.pipeline_sharded_loading(mesh):
            AutoModelForCausalLM.from_pretrained(ckpt, load_in_4bit = True)

    catch("quantized", quantized)
    catch("unknown_schedule", lambda: ppl.pipeline_sharded_loading(mesh, schedule = "pipedream").__enter__())
    return out


CASES = {"load": _case_load, "refusals": _case_refusals}


def _worker(rank, world, port, case, tmp, kwargs):
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
    dist.init_process_group("gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 180))
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(case, world = 2, **kwargs):
    if _torch_pipelining_error():
        pytest.skip(_torch_pipelining_error())
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [ctx.Process(target = _worker, args = (r, world, port, case, tmp, kwargs)) for r in range(world)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False) for r in range(world)]


@pytest.mark.parametrize(
    "arch,world,schedule,v,tie",
    [
        ("llama", 2, "gpipe", None, None),
        ("qwen3", 2, "1f1b", None, None),
        ("gemma3", 2, "gpipe", None, True),  # tied embedding: read on both ends
        ("llama", 4, "gpipe", None, None),  # middle ranks hold neither the embedding nor the head
        ("llama", 2, "interleaved1f1b", 2, None),
        ("gemma3", 2, "zbvzerobubble", 2, True),  # V: rank 0 holds the first and the last stage
    ],
)
def test_sharded_loading_matches_full_load_then_split(arch, world, schedule, v, tie):
    if schedule in ("zbvzerobubble", "dualpipev"):
        import importlib

        if not importlib.import_module("unsloth.models.pp_schedules").v_schedules_supported():
            pytest.skip(f"V schedules need torch >= 2.12 (refused on {torch.__version__})")
    with tempfile.TemporaryDirectory() as ckpt:
        _save_checkpoint(arch, ckpt, tie = tie)
        n_mb = 4 if schedule != "gpipe" or world == 4 else 2
        results = _spawn("load", world, ckpt = ckpt, schedule = schedule, n_mb = n_mb, v = v)
    total_numel = None
    for rank, res in enumerate(results):
        stats = res["stats"]
        assert res["models_loaded"] == 1
        assert tuple(stats["local_stages"]) == tuple(res["local"]) and stats["bounds"] == [tuple(b) for b in res["bounds"]]
        n = stats["loaded_numel"] + stats["placeholder_numel"]
        total_numel = total_numel or n
        assert n == total_numel  # every rank saw the whole model, read part of it
        assert stats["placeholder_numel"] > 0 and stats["read_bytes"] == stats["loaded_numel"] * 4, stats
        # Exactly what this rank's stage keeps after a full load is split: nothing more was read.
        assert stats["loaded_numel"] == res["kept_numel"], (rank, stats["loaded_numel"], res["kept_numel"])
        # Load peak: only this rank's share of the weights exists, and after the split it is what a full load
        # leaves (same stage).
        assert res["bytes"]["loaded"] < res["bytes"]["full"], res["bytes"]
        assert res["bytes"]["after"] == res["bytes"]["full_after"], res["bytes"]
        assert res["unsplit"] and "call apply_pipeline_parallel" in res["unsplit"], res["unsplit"]
        assert res["lora_equal"], "LoRA init differs from the full load"
        assert res["guards"] == 0
        assert res["diffs"] == [], res["diffs"][:6]
        full_losses, full_grads = res["full"]
        losses, grads = res["sharded"]
        assert losses == full_losses, (losses, full_losses)  # bitwise: same weights, same stage program
        assert set(grads) == set(full_grads)
        for k, g in grads.items():
            assert torch.equal(g, full_grads[k]), k


def test_sharded_loading_refusals_leave_the_model_uncut():
    with tempfile.TemporaryDirectory() as ckpt:
        _save_checkpoint("llama", ckpt)
        res = _spawn("refusals", 2, ckpt = ckpt)
    for r in res:
        assert "loaded with pipeline_sharded_loading for another split" in (r["other_split"] or ""), r
        assert r["other_split:uncut"]
        assert "another split" in (r["other_layers_per_stage"] or ""), r
        assert "16-bit safetensors checkpoints only" in (r["quantized"] or ""), r
        assert "pipeline schedule must be one of" in (r["unknown_schedule"] or ""), r
    # Rank 0's head is a placeholder, rank 1's is real: both refuse together (no rank enters the split alone).
    assert "lm_head" in (res[0]["trainable_placeholder_head"] or ""), res[0]
    for r in res:
        assert "trainable but not loaded on every rank" in (r["trainable_placeholder_head"] or ""), r
        assert r["trainable_placeholder_head:uncut"]
