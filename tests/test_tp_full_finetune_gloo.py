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

    def load(
        model_name,
        load_in_4bit = True,
        full_finetuning = False,
        **kwargs,
    ):
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
        out_dir,
        2,
        max_steps = 3,
        adam_epsilon = _ADAM_EPS,
        learning_rate = 1e-3,
        save_strategy = "steps",
        save_steps = 2,
    )
    try:
        trainer = SequentialTrainer(model = model, args = args, train_dataset = data)
    except RuntimeError as e:
        if "full finetuning on a tensor parallel x native FSDP2" not in str(e):
            raise
        return {"refused": None, "trainer_refused": str(e)}
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


def _qat_config(kind):
    import torch as _t
    from torchao.quantization import (
        Int4WeightOnlyConfig,
        Int8DynamicActivationIntxWeightConfig,
        IntxWeightOnlyConfig,
    )
    from torchao.quantization.granularity import PerAxis, PerGroup

    return {
        "int4": lambda: Int4WeightOnlyConfig(group_size = 16),
        "int8_group": lambda: IntxWeightOnlyConfig(weight_dtype = _t.int8, granularity = PerGroup(16)),
        "int8_axis": lambda: IntxWeightOnlyConfig(weight_dtype = _t.int8, granularity = PerAxis(0)),
        "int8_int4": lambda: Int8DynamicActivationIntxWeightConfig(
            weight_dtype = _t.int4, weight_granularity = PerGroup(16)
        ),
    }[kind]()


def _case_qat(
    rank,
    world,
    tmp,
    kind,
    lora,
    kv_heads = 2,
):
    """QAT (torchao QATConfig prepare, as Unsloth's qat_scheme) on full finetuning or under LoRA."""
    from peft import LoraConfig, get_peft_model
    from torch.distributed.device_mesh import init_device_mesh
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig

    from unsloth.models import tp_native as tpn

    repo, overrides = ARCHS["llama"]
    overrides = {
        **overrides,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": kv_heads,
        "head_dim": 8,
    }

    def make():
        model = build(repo, overrides, lora = False)
        decoder = model.model.layers
        quantize_(decoder, QATConfig(_qat_config(kind), step = "prepare"))
        if lora:
            torch.manual_seed(1)
            config = LoraConfig(
                r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False
            )
            model = get_peft_model(model, config)
        else:
            model.requires_grad_(True)
        return model

    ref, model = make(), make()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    try:
        tpn.apply_tensor_parallel(model, mesh["tp"])
    except ValueError as e:
        return {"refused": str(e)}
    fq = sum(type(m).__name__ == "FakeQuantizedLinear" for m in model.modules())
    ids = _batch(ref)
    return {
        "refused": None,
        "fake_quantized": fq,
        "ref": _full_ft_train(ref, ids, 3, _clip_fp64),
        "tp": _full_ft_train(model, ids, 3, tpn.tp_clip_grad_norm_),
    }


CASES = {"api": _case_api, "env": _case_env, "qat": _case_qat}


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

    if case in ("api", "qat"):
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


def _spawn(
    case,
    world,
    env = None,
    tmp = None,
    **kwargs,
):
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
        assert res["losses"] == pytest.approx(ref["losses"], rel = 1e-5), (
            res["losses"],
            ref["losses"],
        )
        assert res["grad_norms"] == pytest.approx(ref["grad_norms"], rel = 1e-4)
        _assert_state(res["state"], ref["state"], f"rank {rank}")
        assert "unsloth_tp" in res["ckpt_files"] and "model.safetensors" in res["ckpt_files"]
    saved = results[0]["saved"]
    assert set(saved) == set(ref["saved"])
    for key, value in ref["saved"].items():
        torch.testing.assert_close(saved[key], value, rtol = 1e-4, atol = 2e-6, msg = key)


def test_env_tp2_x_fsdp2_full_finetuning_is_refused_before_training():
    # tp x native FSDP2 cannot gather the whole model for save_model / Trainer checkpoints: before the fix the
    # run trained and then died at its first checkpoint (save_model refused, pointing at a missing adapter).
    if _torch_below_2_7():
        pytest.skip("torch < 2.7: tensor parallel x FSDP2 is refused at load")
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _save_checkpoint("llama", os.path.join(tmp, "ckpt"))
        results = _spawn(
            "env", 4, {"UNSLOTH_TP_SIZE": "2", "UNSLOTH_FSDP2": "1"}, tmp = tmp, ckpt = ckpt
        )
    for res in results:
        assert res["refused"] is None, res["refused"]  # the load itself is fine
        assert "UNSLOTH_TP_SIZE without UNSLOTH_FSDP2" in res.get("trainer_refused", ""), res


def _torch_below_2_7():
    from packaging.version import Version
    return Version(torch.__version__.split("+")[0]) < Version("2.7")


def _torchao():
    try:
        from torchao.quantization.qat import QATConfig  # noqa: F401
    except Exception:
        return False
    return True


def _skip_unsupported_qat(kind):
    """Older torchao (0.13) cannot QATConfig-prepare every scheme (IntxWeightOnlyConfig, ...): skip there."""
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig

    try:
        quantize_(
            torch.nn.Sequential(torch.nn.Linear(32, 32)),
            QATConfig(_qat_config(kind), step = "prepare"),
        )
    except Exception as e:
        pytest.skip(f"this torchao cannot QAT-prepare {kind}: {type(e).__name__}: {e}")


@pytest.mark.skipif(not _torchao(), reason = "torchao QAT not installed")
@pytest.mark.parametrize(
    "kind,lora,kv_heads",
    [
        ("int4", False, 2),
        ("int8_group", False, 2),
        ("int4", True, 2),
        ("int8_group", True, 2),
        # num_key_value_heads < tp: the replicated key / value projections fake quantize whole rows.
        ("int4", True, 1),
        ("int4", False, 1),
    ],
)
def test_tp2_qat_weight_only_matches_one_process(kind, lora, kv_heads):
    _skip_unsupported_qat(kind)
    with tempfile.TemporaryDirectory() as tmp:
        results = _spawn("qat", 2, tmp = tmp, kind = kind, lora = lora, kv_heads = kv_heads)
    for rank, res in enumerate(results):
        assert res["refused"] is None, res[
            "refused"
        ]  # main: "does not support QAT fake quantization"
        assert res["fake_quantized"] > 0
        ref_grads, ref_losses, ref_norms = res["ref"]
        grads, losses, norms = res["tp"]
        assert set(grads) == set(ref_grads) and ref_grads
        for name in ref_grads:
            torch.testing.assert_close(grads[name], ref_grads[name], rtol = 1e-4, atol = 1e-6, msg = name)
        # Step 1 is exact; afterwards a weight that differs by fp32 reduction-order noise can round to the
        # neighbouring quantization level (a step function), which shows up as ~1e-4 in the later losses.
        assert losses[0] == pytest.approx(ref_losses[0], rel = 1e-6), (rank, losses, ref_losses)
        assert norms[0] == pytest.approx(ref_norms[0], rel = 1e-5), (rank, norms, ref_norms)
        assert losses == pytest.approx(ref_losses, rel = 5e-4), (rank, losses, ref_losses)
        assert norms == pytest.approx(ref_norms, rel = 5e-3), (rank, norms, ref_norms)


@pytest.mark.skipif(not _torchao(), reason = "torchao QAT not installed")
@pytest.mark.parametrize(
    "kind,needle",
    [
        ("int8_axis", "per-row (PerAxis) scale spans the in_features"),
        ("int8_int4", "QAT fake quantization of activations"),
    ],
)
def test_tp2_qat_inexact_schemes_are_refused(kind, needle):
    _skip_unsupported_qat(kind)
    with tempfile.TemporaryDirectory() as tmp:
        results = _spawn("qat", 2, tmp = tmp, kind = kind, lora = False)
    for res in results:
        assert needle in (res["refused"] or ""), res["refused"]
