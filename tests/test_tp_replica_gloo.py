# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Post-training cells on a tensor parallel model (models/tp_replica.py), on a gloo CPU world vs one process:
for_inference / generate on a gathered replica, adapter / tokenizer saves written by rank 0 while the others wait,
merged exports on rank 0, and the sharded model put back for more training."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_gloo import (  # noqa: E402
    _free_port,
    _import_checkout_unsloth,
    _qlora_model,
    _raised,
    _save_checkpoint,
    _train,
)
from test_tp_native_oracle import ARCHS, LLAMA, _batch, build, skip_unsupported  # noqa: E402


def _mesh(world):
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))["tp"]


def _generate(model, prompt, **kwargs):
    kwargs.setdefault("max_new_tokens", 16)
    kwargs.setdefault("pad_token_id", 0)
    return model.generate(input_ids = prompt, **kwargs).tolist()


def _all_plain(model):
    from torch.distributed.tensor import DTensor

    return not any(isinstance(p, DTensor) for p in model.parameters())


def _case_generate(rank, world, tmp, arch):
    """Train 2 steps under tp, then for_inference + greedy / sampled generate on the replica vs one process with
    the same (gathered) weights; for_training (and a training forward) puts the shards back."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    from unsloth import FastLanguageModel
    from unsloth.models import tp_native, tp_replica

    repo, overrides = ARCHS[arch]
    model, ref = build(repo, overrides), build(repo, overrides)
    ids = _batch(model)
    tp_native.apply_tensor_parallel(model, _mesh(world))
    _train(model, ids, 2, tp_native.tp_clip_grad_norm_)
    sharded = dict(model.named_parameters())
    with torch.no_grad():
        loss_before = float(model(input_ids = ids, labels = ids, use_cache = False).loss)
    ref.load_state_dict(tp_native.tp_full_state_dict(model))
    out = {}
    prompt = ids[:1, :5]
    FastLanguageModel.for_inference(model)
    FastLanguageModel.for_inference(ref)
    out["gathered_plain"] = _all_plain(model)
    with torch.no_grad():
        out["logits_err"] = float(
            (model(input_ids = ids).logits - ref(input_ids = ids).logits).abs().max()
        )
    out["greedy_tp"] = _generate(model, prompt, do_sample = False)
    out["greedy_ref"] = _generate(ref, prompt, do_sample = False)
    out["greedy_batch_tp"] = _generate(model, ids[:, :4], do_sample = False, max_new_tokens = 8)
    out["greedy_batch_ref"] = _generate(ref, ids[:, :4], do_sample = False, max_new_tokens = 8)
    torch.manual_seed(100 + rank)  # different on every rank: generate copies tp rank 0's RNG state
    out["sample_tp"] = _generate(model, prompt, do_sample = True, top_k = 0, temperature = 1.5)
    torch.manual_seed(100)
    out["sample_ref"] = _generate(ref, prompt, do_sample = True, top_k = 0, temperature = 1.5)
    FastLanguageModel.for_training(model)
    out["restored"] = all(p is sharded[n] for n, p in model.named_parameters())
    out["is_tp"] = tp_native.is_tensor_parallel(model)
    with torch.no_grad():
        out["loss_err"] = abs(
            float(model(input_ids = ids, labels = ids, use_cache = False).loss) - loss_before
        )
    # generate with no for_inference, then a training forward: the replica is released before it runs.
    _generate(model, prompt, do_sample = False, max_new_tokens = 2)
    out["replica_after_generate"] = tp_replica.replica_active(model)
    model.train()
    loss = model(input_ids = ids, labels = ids, use_cache = False).loss
    loss.backward()
    out["released_by_forward"] = not tp_replica.replica_active(model)
    out["restored_2"] = all(p is sharded[n] for n, p in model.named_parameters())
    out["loss_err_2"] = abs(float(loss) - loss_before)
    out["dtensor_grads"] = all(
        p.grad is not None for n, p in model.named_parameters() if p.requires_grad
    )
    return out


def _case_qlora(rank, world, tmp, nested):
    """4-bit: the gathered weight is the one-process codes bitwise, its dequantized weight equal, same greedy text."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import bitsandbytes as bnb
    from unsloth.models import tp_native, tp_replica

    ref, model = _qlora_model(nested), _qlora_model(nested)
    ref_w = {
        n: (
            m.weight.data.reshape(-1).view(torch.uint8).clone(),
            bnb.functional.dequantize_4bit(m.weight.data, m.weight.quant_state),
        )
        for n, m in ref.named_modules()
        if type(m).__name__ == "Linear4bit"
    }
    tp_native.apply_tensor_parallel(model, _mesh(world))
    ids = _batch(ref)
    prompt = ids[:1, :5]
    with torch.no_grad():
        sharded_logits = model(input_ids = ids).logits
    tp_replica.gather_replica(model)
    codes_equal, deq_equal, shapes = True, True, True
    for n, m in model.named_modules():
        if type(m).__name__ != "Linear4bit":
            continue
        codes, deq = ref_w[n]
        got = m.weight.data.reshape(-1).view(torch.uint8)
        codes_equal &= torch.equal(got, codes)
        deq_equal &= torch.equal(bnb.functional.dequantize_4bit(m.weight.data, m.weight.quant_state), deq)
        shapes &= tuple(m.weight.quant_state.shape) == tuple(deq.shape)
        shapes &= m.weight.quant_state.nested == nested  # the 1-GPU QuantState, double quantized or not
    with torch.no_grad():
        logits = model(input_ids = ids).logits
        ref_logits = ref(input_ids = ids).logits
    out = {
        "codes_equal": codes_equal,
        "deq_equal": deq_equal,
        "shapes": shapes,
        "logits_err": float((logits - ref_logits).abs().max()),
        "sharded_logits_err": float((sharded_logits - ref_logits).abs().max()),
        "greedy_tp": _generate(model, prompt, do_sample = False),
        "greedy_ref": _generate(ref, prompt, do_sample = False),
    }
    tp_replica.release_replica(model)
    with torch.no_grad():
        out["released_logits_err"] = float((model(input_ids = ids).logits - sharded_logits).abs().max())
    return out


def _files(directory):
    out = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            out[name] = open(path, "rb").read()
    return out


def _slow_rank0(obj, attr, seconds = 1.5):
    """Rank 0's writer sleeps first: a rank that returned before rank 0 wrote would see no files."""
    import torch.distributed as dist

    original = getattr(obj, attr)
    if dist.get_rank() != 0:
        return

    def slow(*args, **kwargs):
        time.sleep(seconds)
        return original(*args, **kwargs)

    setattr(obj, attr, slow)


def _case_save(rank, world, tmp):
    """model.save_pretrained / tokenizer.save_pretrained on every rank: rank 0 writes the one-process files,
    the others return after it did, with its return value. Errors on rank 0 reach every rank."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from unsloth.models import tp_native, tp_replica
    from unsloth.save import patch_saving_functions

    model, ref = build(LLAMA), build(LLAMA)
    ids = _batch(model)
    tp_native.apply_tensor_parallel(model, _mesh(world))
    _train(model, ids, 2, tp_native.tp_clip_grad_norm_)
    ref.load_state_dict(tp_native.tp_full_state_dict(model))
    tokenizer = AutoTokenizer.from_pretrained(LLAMA)
    patch_saving_functions(model)
    patch_saving_functions(tokenizer)
    tokenizer.__dict__["_unsloth_tp_world_save"] = True  # what tp_env / apply_tensor_parallel mark
    out = {}
    tp_dir, ref_dir = os.path.join(tmp, "tp"), os.path.join(tmp, "ref")
    _slow_rank0(model, "original_model_save_pretrained")
    out["return"] = model.save_pretrained(tp_dir)
    out["files_on_return"] = os.path.isfile(os.path.join(tp_dir, "adapter_model.safetensors"))
    _slow_rank0(tokenizer, "original_save_pretrained")
    out["tok_return"] = tokenizer.save_pretrained(tp_dir)
    out["tok_files_on_return"] = os.path.isfile(os.path.join(tp_dir, "tokenizer_config.json"))
    out["still_sharded"] = not _all_plain(model) and not tp_replica.replica_active(model)
    if rank == 0:
        ref_tok = AutoTokenizer.from_pretrained(LLAMA)
        patch_saving_functions(ref)
        patch_saving_functions(ref_tok)
        ref.save_pretrained(ref_dir)
        ref_tok.save_pretrained(ref_dir)
    dist.barrier()
    out["files_tp"], out["files_ref"] = _files(tp_dir), _files(ref_dir)
    # Same return on every rank, rank 0's error on every rank.
    out["collective_return"] = tp_replica.tp_collective(
        model, "probe", lambda: {"written_by": dist.get_rank()}, "adapters"
    )

    def fail():
        raise ValueError("disk full")

    out["collective_error"] = _raised(lambda: tp_replica.tp_collective(model, "probe", fail, "adapters"))
    # Reload: every rank loads what rank 0 wrote (the adapter matches the trained tp model).
    from peft import PeftModel

    base = build(LLAMA, lora = False)
    reloaded = PeftModel.from_pretrained(base, tp_dir)
    with torch.no_grad():
        out["reload_err"] = float(
            (reloaded(input_ids = ids).logits - ref(input_ids = ids).logits).abs().max()
        )
    return out


def _case_merged(rank, world, tmp, ckpt):
    """save_pretrained_merged (merged_16bit) on every rank: rank 0 writes what one process writes."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import unsloth_zoo.saving_utils as zoo_saving
    from unsloth.models import tp_native
    from unsloth.save import patch_saving_functions

    zoo_saving._active_merge_device = lambda: "cpu"  # the device spoof says cuda; these ranks have none
    zoo_saving.DEVICE_TYPE = "cpu"

    def load():
        torch.manual_seed(0)
        base = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype = torch.float32)
        torch.manual_seed(1)
        config = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)
        return get_peft_model(base, config)

    model, ref = load(), load()
    tp_native.apply_tensor_parallel(model, _mesh(world))
    tokenizer = AutoTokenizer.from_pretrained(LLAMA)
    patch_saving_functions(model)
    tp_dir, ref_dir = os.path.join(tmp, "merged_tp"), os.path.join(tmp, "merged_ref")
    out = {}
    out["error"] = _raised(
        lambda: model.save_pretrained_merged(tp_dir, tokenizer, save_method = "merged_16bit")
    )
    out["files_on_return"] = os.path.isdir(tp_dir) and any(
        f.endswith(".safetensors") for f in os.listdir(tp_dir)
    )
    out["still_sharded"] = not _all_plain(model)
    if rank == 0:
        patch_saving_functions(ref)
        out["ref_error"] = _raised(
            lambda: ref.save_pretrained_merged(ref_dir, tokenizer, save_method = "merged_16bit")
        )
    dist.barrier()

    def tensors(directory):
        state = {}
        for name in sorted(os.listdir(directory)):
            if name.endswith(".safetensors"):
                state.update(load_file(os.path.join(directory, name)))
        return state

    if out["error"] is None:
        out["tp"], out["ref"] = tensors(tp_dir), tensors(ref_dir)
    return out


def _case_refuse(rank, world, tmp):
    """A replica that does not fit one rank's memory is refused on every rank, with the numbers; PP still is."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    from unsloth import FastLanguageModel
    from unsloth.models import pp_native, tp_native, tp_replica

    model = build(LLAMA)
    tp_native.apply_tensor_parallel(model, _mesh(world))
    sharded = dict(model.named_parameters())
    real = tp_replica._free_bytes
    if rank == 1:
        tp_replica._free_bytes = lambda device: (1 << 10, "free on test")
    out = {"for_inference": _raised(lambda: FastLanguageModel.for_inference(model))}
    out["generate"] = _raised(lambda: model.generate(input_ids = _batch(model)[:1, :4], max_new_tokens = 2))
    tp_replica._free_bytes = real
    out["untouched"] = all(p is sharded[n] for n, p in model.named_parameters())
    out["replica"] = tp_replica.replica_active(model)
    pp = build(LLAMA)
    pp_native.apply_pipeline_parallel(pp, _mesh(world), n_microbatches = 2)
    out["pp_for_inference"] = _raised(lambda: FastLanguageModel.for_inference(pp))
    return out


CASES = {
    "generate": _case_generate,
    "qlora": _case_qlora,
    "save": _case_save,
    "merged": _case_merged,
    "refuse": _case_refuse,
}


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
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(case, world = 2, **kwargs):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        if case == "merged":
            kwargs["ckpt"] = _save_checkpoint("llama", os.path.join(tmp, "ckpt"))
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    return results


@pytest.mark.parametrize("arch", ["llama", "llama_bias", "qwen3", "phi3", "gemma2"])
def test_tp2_generate_on_replica_equals_one_process(arch):
    skip_unsupported(arch)
    for rank, res in enumerate(_spawn("generate", 2, arch = arch)):
        assert res["gathered_plain"], rank
        assert res["logits_err"] == 0.0, (rank, res["logits_err"])
        assert res["greedy_tp"] == res["greedy_ref"], (rank, res["greedy_tp"], res["greedy_ref"])
        assert res["greedy_batch_tp"] == res["greedy_batch_ref"], rank
        # tp rank 0's RNG state on every tp rank: the sampled text is rank 0's, which is one process's.
        assert res["sample_tp"] == res["sample_ref"], (rank, res["sample_tp"], res["sample_ref"])
        assert res["restored"] and res["is_tp"] and res["loss_err"] == 0.0, res
        assert res["replica_after_generate"] and res["released_by_forward"] and res["restored_2"], res
        assert res["loss_err_2"] == 0.0 and res["dtensor_grads"], res


@pytest.mark.parametrize("nested", [False, True])
def test_tp2_qlora_replica_is_the_one_process_weight(nested):
    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        pytest.skip("bitsandbytes not installed")
    for rank, res in enumerate(_spawn("qlora", 2, nested = nested)):
        assert res["codes_equal"] and res["deq_equal"] and res["shapes"], (rank, res)
        assert res["logits_err"] <= 1e-5, (rank, res["logits_err"], res["sharded_logits_err"])
        assert res["greedy_tp"] == res["greedy_ref"], (rank, res)
        assert res["released_logits_err"] == 0.0, res


def test_tp2_adapter_and_tokenizer_saves_equal_one_process():
    results = _spawn("save", 2)
    for rank, res in enumerate(results):
        assert res["return"] is None, res["return"]
        assert res["files_on_return"] and res["tok_files_on_return"], (rank, res)
        assert res["still_sharded"], rank
        assert res["files_tp"].keys() == res["files_ref"].keys(), (
            sorted(res["files_tp"]),
            sorted(res["files_ref"]),
        )
        for name in res["files_ref"]:
            assert res["files_tp"][name] == res["files_ref"][name], name
        assert res["collective_return"] == {"written_by": 0}, (rank, res["collective_return"])
        assert res["collective_error"] and "disk full" in res["collective_error"], res
        assert res["reload_err"] == 0.0, res["reload_err"]
    assert results[0]["tok_return"] == results[1]["tok_return"]
    assert "adapter_model.safetensors" in results[0]["files_ref"]


def test_tp2_merged_save_equals_one_process():
    results = _spawn("merged", 2)
    ref = results[0]
    assert ref["ref_error"] is None, ref["ref_error"]
    for rank, res in enumerate(results):
        assert res["error"] is None, (rank, res["error"])
        assert res["files_on_return"] and res["still_sharded"], (rank, res)
    assert ref["tp"].keys() == ref["ref"].keys(), sorted(set(ref["tp"]) ^ set(ref["ref"]))
    for key, value in ref["ref"].items():
        assert ref["tp"][key].dtype == value.dtype, key
        assert torch.equal(ref["tp"][key], value), key


def test_tp2_replica_that_does_not_fit_is_refused_on_every_rank():
    for rank, res in enumerate(_spawn("refuse", 2)):
        for key in ("for_inference", "generate"):
            assert res[key] and "gathers a whole single-GPU replica" in res[key], (rank, key, res[key])
            assert "GiB" in res[key], res[key]
        assert res["untouched"] and not res["replica"], res
        assert res["pp_for_inference"] and "pipeline parallel" in res["pp_for_inference"], res
