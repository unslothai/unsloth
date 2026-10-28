# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Full finetuning (every base weight trainable, no LoRA) under tensor parallel on a gloo CPU world vs one
process: the direct API (apply_tensor_parallel + a custom loop) and UNSLOTH_TP_SIZE=2 with the HF Trainer
(loader flag full_finetuning = True, tensor parallel applied at Trainer init, Trainer.save_model)."""

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
from test_tp_native_gloo import (  # noqa: E402
    _Rows,
    _clip_fp64,
    _free_port,
    _grads,
    _import_checkout_unsloth,
    _save_checkpoint,
    _trainer_args,
)
from test_tp_native_oracle import ARCHS, _batch, build  # noqa: E402

_ENV_KEYS = ("UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "UNSLOTH_FSDP2", "UNSLOTH_FSDP2_BACKEND")
# Adam's m / (sqrt(v) + eps) turns fp32 reduction-order noise on near-zero gradients into visible weight
# drift at eps 1e-8; at 1e-4 a real tp error (a missing all-reduce) still moves the weights far more.
_ADAM_EPS = 1e-4


def _full_ft_train(model, ids, steps, clip):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-3, eps = _ADAM_EPS, foreach = False)
    losses, norms, grads = [], [], None
    for step in range(steps):
        out = model(input_ids = ids, labels = ids, use_cache = False)
        out.loss.backward()
        if step == 0:
            grads = _grads(model)
        norms.append(float(clip(params, 1.0)))
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss))
    return grads, losses, norms


def _case_api(rank, world, tmp, arch):
    from safetensors.torch import load_file
    from torch.distributed.device_mesh import init_device_mesh

    from unsloth.models import tp_native as tpn

    repo, overrides = ARCHS[arch]
    ref = build(repo, overrides, lora = False)
    model = build(repo, overrides, lora = False)
    for m in (ref, model):
        m.requires_grad_(True)
        m.generation_config.pad_token_id = None  # tiny repos ship -1, which save_pretrained rejects
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    tpn.apply_tensor_parallel(model, mesh["tp"])
    trainable_shards = sum(
        tpn._is_dtensor(p) and p.requires_grad and p.placements[0].is_shard()
        for p in model.parameters()
    )
    ids = _batch(ref)
    ref_out = _full_ft_train(ref, ids, 3, _clip_fp64)
    tp_out = _full_ft_train(model, ids, 3, tpn.tp_clip_grad_norm_)
    final = tpn.tp_full_state_dict(model)
    ref_dir, tp_dir = os.path.join(tmp, "ref"), os.path.join(tmp, "tp")
    if rank == 0:
        ref.save_pretrained(ref_dir)
    # The 1-GPU model, gathered to rank 0 (tp_replica.tp_collective), as Trainer.save_model does.
    from unsloth.models.tp_replica import tp_collective

    tp_collective(model, "save_pretrained", lambda: model.save_pretrained(tp_dir), "all")
    return {
        "ref": ref_out,
        "tp": tp_out,
        "ref_final": {k: v.detach().clone() for k, v in ref.state_dict().items()},
        "tp_final": final,
        "trainable_shards": trainable_shards,
        "saved_ref": load_file(os.path.join(ref_dir, "model.safetensors")) if rank == 0 else None,
        "saved_tp": load_file(os.path.join(tp_dir, "model.safetensors")) if rank == 0 else None,
        "is_tp_after_save": tpn.is_tensor_parallel(model)
        and any(tpn._is_dtensor(p) for p in model.parameters()),
    }


def _case_env(rank, world, tmp, ckpt):
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import transformers
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from unsloth.models import tp_env, tp_native
    from unsloth.models._utils import patch_gradient_accumulation_fix

    patch_gradient_accumulation_fix(transformers.Trainer)

    class SequentialTrainer(transformers.Trainer):
        def _get_train_sampler(self, *args, **kwargs):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    def load(model_name, load_in_4bit = True, full_finetuning = False, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype = torch.float32, attn_implementation = "eager"
        )
        model.requires_grad_(full_finetuning)
        return model

    load = tp_env.tensor_parallel_env_load(load)
    refused = None
    try:
        model = load(ckpt, load_in_4bit = False, full_finetuning = True)
    except Exception as e:
        refused = f"{type(e).__name__}: {e}"
        return {"refused": refused}
    g = torch.Generator().manual_seed(1)
    data = _Rows(list(torch.randint(0, 1000, (3 * 4, 9), generator = g)))
    out_dir = os.path.join(tmp, f"out{world}")
    args = _trainer_args(
        out_dir, 2, max_steps = 3, adam_epsilon = _ADAM_EPS, learning_rate = 1e-3,
        save_strategy = "steps", save_steps = 2,
    )
    trainer = SequentialTrainer(model = model, args = args, train_dataset = data)
    trainer.train()
    is_tp = tp_native.is_tensor_parallel(trainer.model)
    state = (
        tp_native.tp_full_state_dict(trainer.model)
        if is_tp
        else {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
    )
    save_dir = os.path.join(tmp, f"saved{world}")
    trainer.save_model(save_dir)
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    return {
        "refused": refused,
        "is_tp": is_tp,
        "losses": [h["loss"] for h in logs],
        "grad_norms": [h.get("grad_norm") for h in logs],
        "state": state,
        "saved": load_file(os.path.join(save_dir, "model.safetensors")) if rank == 0 else None,
        "ckpt_files": sorted(os.listdir(os.path.join(out_dir, "checkpoint-2"))),
    }


CASES = {"api": _case_api, "env": _case_env}


def _worker(rank, world, port, case, tmp, env, kwargs):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    for key in _ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(env)
    torch.set_num_threads(2)
    import datetime

    import torch.distributed as dist

    if case == "api":
        error = _import_checkout_unsloth()
        if error is not None:
            torch.save({"skip": error}, os.path.join(tmp, f"result_{rank}.pt"))
            return
        dist.init_process_group(
            "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 300)
        )
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn(case, world, env = None, tmp = None, **kwargs):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    procs = [
        ctx.Process(target = _worker, args = (r, world, port, case, tmp, env or {}, kwargs))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(900)
    codes = [p.exitcode for p in procs]
    for p in procs:
        if p.is_alive():
            p.kill()
    assert codes == [0] * world, f"rank exit codes {codes}"
    results = [
        torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False) for r in range(world)
    ]
    for r in range(world):
        os.remove(os.path.join(tmp, f"result_{r}.pt"))
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    return results


def _assert_state(got, want, what):
    assert set(got) == set(want), sorted(set(got) ^ set(want))[:6]
    for key, value in want.items():
        torch.testing.assert_close(got[key], value, rtol = 1e-4, atol = 2e-6, msg = f"{what} {key}")


@pytest.mark.parametrize("arch", ["llama", "llama_bias", "qwen3", "gemma2"])
def test_tp2_full_finetuning_matches_one_process(arch):
    with tempfile.TemporaryDirectory() as tmp:
        results = _spawn("api", 2, tmp = tmp, arch = arch)
    for rank, res in enumerate(results):
        ref_grads, ref_losses, ref_norms = res["ref"]
        grads, losses, norms = res["tp"]
        assert res["trainable_shards"] > 0  # the sharded base weights themselves train
        assert set(grads) == set(ref_grads) and len(ref_grads) > 0
        for name in ref_grads:
            torch.testing.assert_close(grads[name], ref_grads[name], rtol = 1e-4, atol = 1e-6, msg = name)
        assert losses == pytest.approx(ref_losses, rel = 1e-5, abs = 1e-6), (rank, losses, ref_losses)
        assert norms == pytest.approx(ref_norms, rel = 1e-5), (rank, norms, ref_norms)
        _assert_state(res["tp_final"], res["ref_final"], f"rank {rank}")
        assert res["is_tp_after_save"]
    saved_tp, saved_ref = results[0]["saved_tp"], results[0]["saved_ref"]
    assert set(saved_tp) == set(saved_ref)
    for key in saved_ref:
        assert saved_tp[key].shape == saved_ref[key].shape, key
        torch.testing.assert_close(saved_tp[key], saved_ref[key], rtol = 1e-4, atol = 2e-6, msg = key)


def test_env_tp2_full_finetuning_trainer_matches_one_process():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _save_checkpoint("llama", os.path.join(tmp, "ckpt"))
        (ref,) = _spawn("env", 1, tmp = tmp, ckpt = ckpt)
        results = _spawn("env", 2, {"UNSLOTH_TP_SIZE": "2"}, tmp = tmp, ckpt = ckpt)
    assert ref["refused"] is None and not ref["is_tp"]
    for rank, res in enumerate(results):
        assert res["refused"] is None, res["refused"]  # main: "does not support full finetuning"
        assert res["is_tp"]
        assert res["losses"] == pytest.approx(ref["losses"], rel = 1e-5), (res["losses"], ref["losses"])
        assert res["grad_norms"] == pytest.approx(ref["grad_norms"], rel = 1e-4)
        _assert_state(res["state"], ref["state"], f"rank {rank}")
        assert "unsloth_tp" in res["ckpt_files"] and "model.safetensors" in res["ckpt_files"]
    saved = results[0]["saved"]
    assert set(saved) == set(ref["saved"])
    for key, value in ref["saved"].items():
        torch.testing.assert_close(saved[key], value, rtol = 1e-4, atol = 2e-6, msg = key)

