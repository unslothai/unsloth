# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_PP_SIZE (models/pp_env.py): pipeline parallel from the environment inside the HF Trainer, on a gloo
CPU world (pp 2, pp 2 x tp 2, pp 2 x dp 2 DDP-style / native FSDP2), vs the same script in one process.

The workers set the torchrun environment and let tp_env create the process group and the mesh; the loader /
get_peft_model hooks wrap AutoModelForCausalLM / PEFT the way they wrap FastLanguageModel / FastModel."""

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
    _free_port,
    _import_checkout_unsloth,
    _raised,
    _save_checkpoint,
    _trainer_args,
)

_ADAM_EPS = 1e-4  # as test_tp_env_gloo: fp32 reduction-order noise stays below 1e-6 over 3 steps
_ENV_KEYS = (
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_PP_MICROBATCHES",
    "UNSLOTH_PP_SCHEDULE",
    "UNSLOTH_PP_KEEP_REPLICA",
    "UNSLOTH_PP_STAGES_PER_RANK",
    "UNSLOTH_PP_FOLD_GA",
    "UNSLOTH_FSDP2",
    "UNSLOTH_FSDP2_BACKEND",
    "ACCELERATE_USE_FSDP",
)
N_LAYERS = 4


def _checkpoint(tmp, arch = "llama"):
    overrides = {"num_hidden_layers": N_LAYERS}
    if arch == "llama_dynamic_rope":
        # Dynamic NTK RoPE past 4 positions: the frequencies follow each batch's max(position_ids) + 1.
        arch = "llama"
        overrides.update(
            rope_scaling = {"rope_type": "dynamic", "factor": 2.0}, max_position_embeddings = 4
        )
    if (
        arch == "gemma3"
    ):  # sliding / full layers on both stages, window shorter than the 9-token rows
        overrides.update(
            layer_types = ["sliding_attention", "full_attention"] * (N_LAYERS // 2), sliding_window = 4
        )
    return _save_checkpoint(arch, os.path.join(tmp, "ckpt"), overrides)


def _case_units(rank, world, tmp):
    """Layout parsing, load / Trainer refusals (no process group)."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    from unsloth.models import pp_env, tp_env

    out = {}
    # TrainingArguments set up accelerate's state from WORLD_SIZE: build them before the layouts below set it.
    args = _trainer_args(tmp, 2)
    best = _trainer_args(
        tmp, 2, load_best_model_at_end = True, eval_strategy = "steps", save_strategy = "steps"
    )
    smoothed = _trainer_args(tmp, 2, label_smoothing_factor = 0.3)
    try:
        from trl import SFTConfig, SFTTrainer
        packed = SFTConfig(output_dir = tmp, packing = True, report_to = [], use_cpu = True)
        plain_sft = SFTConfig(output_dir = tmp, report_to = [], use_cpu = True)
    except ImportError:
        SFTTrainer = None

    def layout(**env):
        for key in _ENV_KEYS + ("WORLD_SIZE",):
            os.environ.pop(key, None)
        os.environ.update({k: str(v) for k, v in env.items()})
        tp_env._reset_env_layout()
        try:
            got = tp_env.env_layout()
        except Exception as e:
            return f"{type(e).__name__}: {e}"
        return (
            None
            if got is None
            else (
                got.pp,
                got.dp,
                got.tp,
                got.world,
                got.fsdp2,
                got.microbatches,
                got.schedule,
                repr(got),
                got.schedule_auto,
            )
        )

    out["pp2"] = layout(UNSLOTH_PP_SIZE = 2, WORLD_SIZE = 2)
    out["pp2_dp2"] = layout(UNSLOTH_PP_SIZE = 2, WORLD_SIZE = 4)
    out["pp2_fsdp2"] = layout(UNSLOTH_PP_SIZE = 2, UNSLOTH_FSDP2 = 1, WORLD_SIZE = 4)
    out["pp2_tp2"] = layout(UNSLOTH_PP_SIZE = 2, UNSLOTH_TP_SIZE = 2, WORLD_SIZE = 4)
    out["pp2_tp2_dp2"] = layout(UNSLOTH_PP_SIZE = 2, UNSLOTH_TP_SIZE = 2, WORLD_SIZE = 8)
    out["pp_one"] = layout(UNSLOTH_PP_SIZE = 1, WORLD_SIZE = 4)
    out["no_divide"] = layout(UNSLOTH_PP_SIZE = 3, WORLD_SIZE = 4)
    out["no_divide_tp"] = layout(UNSLOTH_PP_SIZE = 2, UNSLOTH_TP_SIZE = 2, WORLD_SIZE = 6)
    out["zero"] = layout(UNSLOTH_PP_SIZE = 0, WORLD_SIZE = 2)
    out["text"] = layout(UNSLOTH_PP_SIZE = "two", WORLD_SIZE = 2)
    out["mb"] = layout(UNSLOTH_PP_SIZE = 2, UNSLOTH_PP_MICROBATCHES = 3, WORLD_SIZE = 2)
    out["mb_text"] = layout(UNSLOTH_PP_SIZE = 2, UNSLOTH_PP_MICROBATCHES = "x", WORLD_SIZE = 2)
    out["schedule"] = layout(UNSLOTH_PP_SIZE = 2, UNSLOTH_PP_SCHEDULE = "1F1B", WORLD_SIZE = 2)
    out["schedule_bad"] = layout(UNSLOTH_PP_SIZE = 2, UNSLOTH_PP_SCHEDULE = "zb", WORLD_SIZE = 2)
    out["schedule_gpipe"] = layout(UNSLOTH_PP_SIZE = 2, UNSLOTH_PP_SCHEDULE = "gpipe", WORLD_SIZE = 2)
    out["default_mb1"] = layout(UNSLOTH_PP_SIZE = 2, UNSLOTH_PP_MICROBATCHES = 1, WORLD_SIZE = 2)
    out["schedule_mb"] = layout(
        UNSLOTH_PP_SIZE = 4, UNSLOTH_PP_SCHEDULE = "1f1b", UNSLOTH_PP_MICROBATCHES = 2, WORLD_SIZE = 4
    )
    out["accelerate_fsdp"] = layout(
        UNSLOTH_PP_SIZE = 2, UNSLOTH_FSDP2 = 1, UNSLOTH_FSDP2_BACKEND = "accelerate", WORLD_SIZE = 4
    )

    calls = []

    def loader(
        model_name,
        load_in_4bit = True,
        fast_inference = False,
        full_finetuning = False,
        **kw,
    ):
        calls.append(model_name)
        return "loaded"

    layout(UNSLOTH_PP_SIZE = 2, WORLD_SIZE = 2)
    wrapped = tp_env.tensor_parallel_env_load(loader)
    out["refuse_vllm"] = _raised(lambda: wrapped("m", fast_inference = True))
    out["refuse_full_ft"] = _raised(lambda: wrapped("m", False, full_finetuning = True))
    out["refused_before_load"] = calls == []

    # Trainer refusals (classes / arguments), before anything is split.
    import transformers

    lay = tp_env.env_layout()

    class DPOTrainer(transformers.Trainer):
        pass

    DPOTrainer.__module__ = "trl.trainer.dpo_trainer"

    class UnslothGRPOTrainer(transformers.Trainer):
        pass

    class MyTrainer(transformers.Trainer):
        def compute_loss(self, *args, **kwargs):
            return None

    class PlainSubclass(transformers.Trainer):
        def _get_train_sampler(self, *args, **kwargs):
            return None

    out["dpo"] = _raised(lambda: pp_env.pp_trainer_refusals(DPOTrainer, args, lay))
    out["grpo"] = _raised(lambda: pp_env.pp_trainer_refusals(UnslothGRPOTrainer, args, lay))
    out["custom_loss"] = _raised(lambda: pp_env.pp_trainer_refusals(MyTrainer, args, lay))
    out["plain"] = _raised(lambda: pp_env.pp_trainer_refusals(PlainSubclass, args, lay))
    out["trainer"] = _raised(lambda: pp_env.pp_trainer_refusals(transformers.Trainer, args, lay))
    out["best"] = _raised(lambda: pp_env.pp_trainer_refusals(transformers.Trainer, best, lay))
    # Failing before: label smoothing and compute_loss_func were accepted and silently skipped.
    out["smoothing"] = _raised(
        lambda: pp_env.pp_trainer_refusals(transformers.Trainer, smoothed, lay)
    )
    out["loss_func"] = _raised(
        lambda: pp_env.pp_trainer_refusals(
            transformers.Trainer, args, lay, compute_loss_func = lambda *a, **k: None
        )
    )
    if SFTTrainer is not None:
        out["packing"] = _raised(lambda: pp_env.pp_trainer_refusals(SFTTrainer, packed, lay))
        out["sft"] = _raised(lambda: pp_env.pp_trainer_refusals(SFTTrainer, plain_sft, lay))
    else:
        out["packing"] = out["sft"] = "no trl"
    # A model the split refuses is refused at from_pretrained, naming the variable: a mixture-of-experts
    # family whose router auxiliary loss pipeline parallel does not carry (Mixtral). Qwen3-MoE, whose loss it
    # carries between stages (tests/test_pp_moe_aux_gloo.py), loads.
    from transformers import AutoConfig, AutoModelForCausalLM

    config = transformers.MixtralConfig(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 96,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        num_local_experts = 4,
        num_experts_per_tok = 2,
    )
    moe = tp_env.tensor_parallel_env_load(
        lambda name, load_in_4bit = True: AutoModelForCausalLM.from_config(config)
    )
    qwen3_moe_config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Qwen3MoeForCausalLM")
    qwen3_moe = tp_env.tensor_parallel_env_load(
        lambda name, load_in_4bit = True: AutoModelForCausalLM.from_config(qwen3_moe_config)
    )
    os.environ.update(
        MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(_free_port()), RANK = "0", LOCAL_RANK = "0"
    )
    layout(UNSLOTH_PP_SIZE = 1, WORLD_SIZE = 1)  # (no world for the mesh here: refusal path only)
    os.environ["UNSLOTH_PP_SIZE"] = "2"
    tp_env._reset_env_layout()
    tp_env._LAYOUT = tp_env.TPEnvLayout(1, 1, 2, False, 2)
    tp_env._LAYOUT._mesh = object()  # the refusal comes from the model, not the mesh
    out["moe"] = _raised(lambda: moe("m", load_in_4bit = False))
    out["qwen3_moe"] = _raised(lambda: qwen3_moe("m", load_in_4bit = False))
    # Failing before: under pp x tp the tensor split in Trainer.__init__ replaced the parameters a prebuilt
    # optimizer held, so training stepped the old ones (no update, no error).
    from unsloth.models import _utils as unsloth_utils

    tp_env._LAYOUT = tp_env.TPEnvLayout(2, 1, 4, False, 2)
    trainer_like = object.__new__(transformers.Trainer)
    prebuilt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr = 0.1)
    out["prebuilt_optimizer"] = _raised(
        lambda: unsloth_utils._pipeline_parallel_env_refusals(
            trainer_like,
            transformers.Trainer.__init__,
            (),
            {"args": args, "optimizers": (prebuilt, None)},
        )
    )
    out["prebuilt_param_groups"] = _raised(
        lambda: unsloth_utils._pipeline_parallel_env_refusals(
            trainer_like,
            transformers.Trainer.__init__,
            (),
            {
                "args": args,
                "optimizer_cls_and_kwargs": (
                    torch.optim.SGD,
                    {"params": prebuilt.param_groups, "lr": 0.1},
                ),
            },
        )
    )
    out["no_prebuilt_optimizer"] = _raised(
        lambda: unsloth_utils._pipeline_parallel_env_refusals(
            trainer_like, transformers.Trainer.__init__, (), {"args": args}
        )
    )
    # Hidden states sent between stages: float32 for a model Unsloth trains in float32 (Gemma 3 on a T4).
    import types

    from unsloth.models import pp_native

    parts = types.SimpleNamespace(
        embed = types.SimpleNamespace(weight = torch.zeros(1, dtype = torch.float16))
    )
    plain, forced = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
    forced._unsloth_forced_float32 = True
    out["act_dtype"] = (
        pp_native._default_activation_dtype(plain, parts),
        pp_native._default_activation_dtype(forced, parts),
    )

    # Later stages' embedding stub returns zeros in the dtype the real (here upcasting) embedding returns.
    class Upcast(torch.nn.Embedding):
        def forward(self, ids):
            return super().forward(ids).float()

    stub = pp_native._PPEmbeddingStub(Upcast(10, 4).half())
    out["stub_dtype"] = stub(torch.zeros(2, 3, dtype = torch.long)).dtype
    return out


def _case_env(
    rank,
    world,
    tmp,
    ckpt,
    bs,
    steps = 3,
    evaluate = True,
    post = True,
    optim = "adamw_torch",
    ga = 2,
    eval_bs = None,
    noisy = False,
    varlen = False,
    mixed_keys = False,
    neftune = None,
):
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import torch.distributed as dist
    import transformers
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    from unsloth.models import pp_env, pp_native, tp_env, tp_native
    from unsloth.models._utils import patch_gradient_accumulation_fix

    patch_gradient_accumulation_fix(transformers.Trainer)
    layout = tp_env.env_layout()
    seen = []

    class SequentialTrainer(transformers.Trainer):
        def _get_train_sampler(self, *args, **kwargs):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    def collate(rows):
        batch = {
            "input_ids": torch.stack([r["input_ids"] for r in rows]),
            "labels": torch.stack([r["labels"] for r in rows]).clone(),
        }
        # Uneven token counts per micro-batch / replica, the same per row however the rows are batched.
        batch["labels"][batch["labels"] % 5 == 0] = -100
        if varlen:
            # Rows of 2 to 9 tokens (fixed per row), right padded to the batch's longest with a mask.
            lengths = [2 + int(r["input_ids"].sum()) % 8 for r in rows]
            width = max(lengths)
            mask = torch.arange(9)[None, :] < torch.tensor(lengths)[:, None]
            batch["input_ids"] = torch.where(mask, batch["input_ids"], 0)[:, :width]
            batch["labels"] = torch.where(mask, batch["labels"], -100)[:, :width]
            batch["attention_mask"] = mask[:, :width].long()
            if mixed_keys and len(seen) % 2:
                del batch[
                    "attention_mask"
                ]  # right padding without a mask: same loss under causal attention
        if noisy:
            # A stochastic collator whose RNG the stages advance differently (as dropout over different layers).
            if rank % 2 == 1:
                torch.rand(7)
            swap = torch.rand(batch["input_ids"].shape) < 0.3
            batch["input_ids"] = torch.where(
                swap, torch.randint_like(batch["input_ids"], 1, 50), batch["input_ids"]
            )
        seen.append(batch["input_ids"].sum().item())
        return transformers.BatchEncoding(
            batch
        )  # what tokenizer-based collators return (a Mapping)

    def load(
        model_name,
        load_in_4bit = True,
        **kwargs,
    ):  # FastLanguageModel.from_pretrained's flags
        return AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype = torch.float32, attn_implementation = "eager"
        )

    def peft(model):
        torch.manual_seed(1)
        config = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)
        return get_peft_model(model, config)

    load = tp_env.tensor_parallel_env_load(load)
    peft = tp_env.tensor_parallel_env_peft(peft)
    model = peft(load(ckpt, load_in_4bit = False))
    pre_split = not pp_native.is_pipeline_parallel(model)
    dp = layout.dp if layout is not None else 1
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 1000, (steps * ga * bs * max(dp, 1), 9), generator = g)
    data = _Rows(list(ids))
    eval_data = _Rows(list(torch.randint(0, 1000, (4 * bs * max(dp, 1), 9), generator = g)))
    args = _trainer_args(
        os.path.join(tmp, "out"),
        bs,
        optim = optim,
        max_steps = steps,
        adam_epsilon = _ADAM_EPS,
        gradient_accumulation_steps = ga,
        per_device_eval_batch_size = eval_bs or bs,
        **({} if neftune is None else {"neftune_noise_alpha": neftune}),
    )
    trainer = SequentialTrainer(model = model, args = args, train_dataset = data, data_collator = collate)
    stage_ids = []
    split = pp_native.pipeline_stage_of(trainer.model)
    if split is not None and not split.is_first:
        # Later stages run the model on the batch's real token ids (not zeros), as one GPU does.
        def record(module, args, kwargs):
            if torch.is_grad_enabled() and kwargs.get("input_ids") is not None:
                stage_ids.append(int(kwargs["input_ids"].sum()))

        split.parts.inner.register_forward_pre_hook(record, with_kwargs = True)
    trainer.train()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    res = {
        "steps": [h["step"] for h in logs],
        "losses": [h["loss"] for h in logs],
        "grad_norms": [h.get("grad_norm") for h in logs],
        "seen": list(seen),
        "stage_ids": list(stage_ids),  # the training run above (not the later cells)
        "pre_split": pre_split,
        "layout": None if layout is None else repr(layout),
        "is_pp": pp_native.is_pipeline_parallel(trainer.model),
        "wrapped": type(trainer.model_wrapped).__name__,
    }
    stage = pp_native.pipeline_stage_of(trainer.model)
    if stage is not None:
        res["stage"] = stage.stage_index
        res["schedule"] = (stage.schedule, stage.schedule_auto)
        # The torch schedules the training steps actually ran (eval / later cells come after this).
        res["step_schedules"] = sorted({type(s).__name__ for s in stage._shapes.values()})
        res["local_stages"] = tuple(getattr(stage, "local_stages", None) or (stage.stage_index,))
        # Gradient accumulation folded into one schedule step per window (pp_env._pp_window_step).
        res["fold"] = dict(stage.__dict__.get("_unsloth_fold_stats") or {})
        # Every stage seeds its train sampler per dp replica (LengthGroupedSampler would draw from the global RNG).
        res["sampler_hooked"] = "_get_train_sampler" in trainer.__dict__
        res["owned"] = sorted(n for n, p in trainer.model.named_parameters() if p.requires_grad)
        res["kept_on_cpu"] = (
            all(p.device.type == "cpu" for _, _, m, _ in stage._kept for p in m.parameters())
            if stage._kept and stage.tp_mesh is None
            else None
        )
    if evaluate:
        res["eval_loss"] = trainer.evaluate(eval_data)["eval_loss"]
    save_dir = os.path.join(tmp, "adapter")
    if stage is not None:
        pp_env.pp_save_adapter(trainer.model, (save_dir,), {})
    elif rank == 0:
        trainer.model.save_pretrained(save_dir)
    if dist.is_initialized():
        dist.barrier()
    res["adapter"] = (
        load_file(os.path.join(save_dir, "adapter_model.safetensors")) if rank == 0 else None
    )
    trainer.save_model(os.path.join(tmp, "trainer_save"))
    if dist.is_initialized():
        dist.barrier()
    res["trainer_adapter"] = (
        load_file(os.path.join(tmp, "trainer_save", "adapter_model.safetensors"))
        if rank == 0
        else None
    )
    if post:
        prompt = ids[:1, :4]
        refusal = pp_env.replica_refusal(stage, "generate") if stage is not None else None
        if refusal is None:
            trainer.model.eval()
            out = trainer.model.generate(
                input_ids = prompt, max_new_tokens = 4, do_sample = False, pad_token_id = 0
            )
            res["generated"] = out.tolist()
            if stage is not None:
                res["replica_active"] = stage.replica is not None
                # Failing before: the other stages' modules came back in training mode (dropout on).
                res["replica_training"] = sorted(
                    n for n, m in trainer.model.named_modules() if m.training
                )[:3]
                # An export: the whole one-GPU state dict, built on rank 0 only (the others wait).
                full = pp_env.pp_collective(
                    trainer.model,
                    "export",
                    lambda: {k: v.detach().clone() for k, v in trainer.model.state_dict().items()},
                )
                res["export"] = full if rank == 0 else None
                # The other ranks return rank 0's result (as tp_replica's collective saves do).
                res["export_other"] = sorted(full) if rank != 0 else None
                # Failing before: a checkpoint written with the replica active recorded every stage's params.
                trainer._save_optimizer_and_scheduler(os.path.join(tmp, "ckpt_with_replica"))
                res["released_by_save"] = stage.replica is None
                trainer.model.train()
                pp_env.release_replica(trainer.model)
                res["released"] = stage.replica is None and all(
                    not isinstance(layer, torch.nn.Module)
                    or type(layer).__name__ == "_PPPassThrough"
                    for i, layer in enumerate(stage.parts.layers)
                    if not (stage.start <= i < stage.end)
                )
                # A training step after generate: the pipeline split is back.
                trainer.args.max_steps = steps + 1
                res["after_generate"] = _raised(lambda: trainer.train(resume_from_checkpoint = None))
            else:
                res["export"] = {
                    k: v.detach().clone() for k, v in trainer.model.state_dict().items()
                }
        else:
            res["generate_refused"] = _raised(
                lambda: trainer.model.generate(
                    input_ids = prompt, max_new_tokens = 4, do_sample = False, pad_token_id = 0
                )
            )
            res["export_refused"] = _raised(
                lambda: pp_env.pp_collective(trainer.model, "export", lambda: None)
            )
    if dist.is_initialized():
        dist.barrier()
    return res


def _case_resume(
    rank,
    world,
    tmp,
    ckpt,
    bs = 2,
):
    """save_strategy = "steps" checkpoints (the 1-GPU adapter + per-rank optimizer state) and an exact resume."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from unsloth.models import tp_env
    from unsloth.models._utils import patch_gradient_accumulation_fix

    patch_gradient_accumulation_fix(transformers.Trainer)
    layout = tp_env.env_layout()
    dp = layout.dp if layout is not None else 1

    class SequentialTrainer(transformers.Trainer):
        def _get_train_sampler(self, *args, **kwargs):
            return torch.utils.data.SequentialSampler(self.train_dataset)

    load = tp_env.tensor_parallel_env_load(
        lambda name, load_in_4bit = True: AutoModelForCausalLM.from_pretrained(
            name, torch_dtype = torch.float32, attn_implementation = "eager"
        )
    )

    def fresh():
        torch.manual_seed(1)
        config = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)
        return get_peft_model(load(ckpt, load_in_4bit = False), config)

    g = torch.Generator().manual_seed(1)
    data = _Rows(list(torch.randint(0, 1000, (4 * 2 * bs * dp, 9), generator = g)))

    def run(out):
        args = _trainer_args(
            os.path.join(tmp, out),
            bs,
            max_steps = 4,
            adam_epsilon = _ADAM_EPS,
            save_strategy = "steps",
            save_steps = 2,
        )
        return SequentialTrainer(model = fresh(), args = args, train_dataset = data)

    def summary(trainer):
        logs = [h for h in trainer.state.log_history if "loss" in h]
        return {
            "steps": [h["step"] for h in logs],
            "losses": [h["loss"] for h in logs],
            "norms": [h["grad_norm"] for h in logs],
        }

    a = run("a")
    a.train()
    ckpt2 = os.path.join(tmp, "a", "checkpoint-2")
    files = sorted(os.listdir(ckpt2)) + sorted(os.listdir(os.path.join(ckpt2, "unsloth_tp")))
    b = run("b")
    b.train(resume_from_checkpoint = ckpt2)
    state_a = {n: p.detach().clone() for n, p in a.model.named_parameters() if p.requires_grad}
    state_b = {n: p.detach().clone() for n, p in b.model.named_parameters() if p.requires_grad}
    return {
        "a": summary(a),
        "b": summary(b),
        "files": files,
        "same": all(torch.equal(state_a[k], state_b[k]) for k in state_a),
    }


def _case_scaler_clip(
    rank,
    world,
    tmp,
    wrapper_unscale = False,
):
    """float16: the Trainer clip under pp unscales the gradients once, for the torch optimizer the
    AcceleratedOptimizer hands the GradScaler (a second unscale inside scaler.step would shrink the update)."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import types

    from accelerate.optimizer import AcceleratedOptimizer
    from torch.distributed.device_mesh import init_device_mesh
    from test_pp_fsdp2_gloo import LLAMA, build, global_batch

    from unsloth.models import pp_env, pp_native

    import torch.distributed as dist

    if wrapper_unscale:  # the old lookup: unscale the AcceleratedOptimizer itself
        pp_env._unwrap_optimizer = lambda opt: opt

    dist.init_process_group("gloo")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))
    model = build(LLAMA)
    stage = pp_native.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = 2)
    ids, labels = global_batch(model, 4)
    from accelerate import Accelerator

    Accelerator(cpu = True)  # the state AcceleratedOptimizer reads
    scaler = torch.amp.GradScaler("cpu", init_scale = 1024.0)
    inner = torch.optim.SGD(stage.parameters(), lr = 1.0)
    optimizer = AcceleratedOptimizer(inner, device_placement = False, scaler = scaler)
    accelerator = types.SimpleNamespace(scaler = scaler, _optimizers = [optimizer])
    stage.step(ids, labels, scaler = scaler)
    norm = pp_env._pp_clip(accelerator, stage)(model.parameters(), 1e9)
    grads = [p.grad.detach().clone() for p in stage.parameters()]
    before = [p.detach().clone() for p in stage.parameters()]
    optimizer.step()
    deltas = [(b - p.detach()) for b, p in zip(before, stage.parameters())]
    return {
        "norm": float(norm),
        "max_rel": max(
            float((d - g).abs().max() / g.abs().max().clamp_min(1e-30))
            for d, g in zip(deltas, grads)
        ),
        "scale": scaler.get_scale(),
    }


def _case_dp_agree(rank, world, tmp):
    """pp_env._dp_agree: folding only when every dp replica folds with the same first-batch rows."""
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import types

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from unsloth.models import pp_env

    dist.init_process_group("gloo")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("dp",))
    stage = types.SimpleNamespace(
        dp_mesh = mesh, ddp_mesh = None, group = None, device = torch.device("cpu")
    )
    return {
        "same": pp_env._dp_agree(stage, True, 2),
        "one_off": pp_env._dp_agree(stage, rank == 0, 2),
        "rows_differ": pp_env._dp_agree(stage, True, 2 if rank == 0 else 1),
    }


CASES = {
    "dp_agree": _case_dp_agree,
    "units": _case_units,
    "env": _case_env,
    "resume": _case_resume,
    "scaler_clip": _case_scaler_clip,
}


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
    import faulthandler

    faulthandler.dump_traceback_later(
        max(int(os.environ.get("PP_TEST_TIMEOUT", 600)) - 30, 30), exit = True
    )
    import torch.distributed as dist

    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn(
    case,
    world = 1,
    env = None,
    **kwargs,
):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        if case in ("env", "resume"):
            kwargs["ckpt"] = _checkpoint(tmp, kwargs.pop("arch", "llama"))
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, env or {}, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(int(os.environ.get("PP_TEST_TIMEOUT", 600)))
        codes = [p.exitcode for p in procs]
        for p in procs:
            if p.is_alive():
                p.kill()
        assert codes == [0] * world, f"rank exit codes {codes}"
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]
    if "skip" in results[0]:
        pytest.skip(f"import unsloth failed on this host: {results[0]['skip']}")
    return results


def _pipelining_available():
    try:
        import inspect

        from torch.distributed.pipelining import ScheduleGPipe
    except ImportError:
        return False
    return "return_outputs" in inspect.signature(ScheduleGPipe.step).parameters


needs_pipelining = pytest.mark.skipif(
    not _pipelining_available(),
    reason = "torch.distributed.pipelining with step(return_outputs) (torch >= 2.10)",
)


def _close(
    got,
    want,
    rel = 1e-5,
    what = "",
):
    assert got["steps"] == want["steps"], (what, got["steps"], want["steps"])
    assert got["losses"] == pytest.approx(want["losses"], rel = rel), (
        what,
        got["losses"],
        want["losses"],
    )
    # Step 1 is tight; later steps carry the one-process reference's own run-to-run spread on this host
    # (3e-5 absolute at a 2.3 grad norm between two identical reference runs).
    assert got["grad_norms"][:1] == pytest.approx(want["grad_norms"][:1], rel = rel), (
        what,
        got["grad_norms"],
    )
    assert got["grad_norms"] == pytest.approx(want["grad_norms"], rel = 1e-4), (
        what,
        got["grad_norms"],
        want["grad_norms"],
    )
    if "eval_loss" in want:
        assert got["eval_loss"] == pytest.approx(want["eval_loss"], rel = rel), (
            what,
            got["eval_loss"],
            want["eval_loss"],
        )


def _same_tensors(
    got,
    want,
    rel = 1e-4,
    atol = 1e-5,
    what = "",
):
    assert set(got) == set(want) and want, (what, sorted(set(got) ^ set(want))[:6])
    for key, value in want.items():
        torch.testing.assert_close(
            got[key], value, rtol = rel, atol = atol, msg = lambda m: f"{what} {key}: {m}"
        )


def _ref(bs, **kwargs):
    return _spawn("env", 1, bs = bs, **kwargs)[0]


@needs_pipelining
def test_pp_env_layout_and_refusals():
    (res,) = _spawn("units")
    assert res["pp2"][:5] == (2, 1, 1, 2, False) and res["pp2"][7] == "pp 2", res["pp2"]
    assert res["pp2_dp2"][:5] == (2, 2, 1, 4, False) and res["pp2_dp2"][7] == "pp 2 x dp 2 (DDP)"
    assert res["pp2_fsdp2"][:5] == (2, 2, 1, 4, True) and "native FSDP2" in res["pp2_fsdp2"][7]
    assert res["pp2_tp2"][:5] == (2, 1, 2, 4, False) and res["pp2_tp2"][7] == "pp 2 x tp 2"
    assert res["pp2_tp2_dp2"][:5] == (2, 2, 2, 8, False)
    assert res["pp_one"] is None
    assert "UNSLOTH_PP_SIZE=3" in res["no_divide"] and "WORLD_SIZE=4" in res["no_divide"], res[
        "no_divide"
    ]
    assert "UNSLOTH_PP_SIZE x UNSLOTH_TP_SIZE = 4" in res["no_divide_tp"], res["no_divide_tp"]
    assert "must be >= 1" in res["zero"] and "must be an integer" in res["text"]
    assert res["mb"][5] == 3 and "UNSLOTH_PP_MICROBATCHES" in res["mb_text"]
    assert res["schedule"][6] == "1f1b" and "UNSLOTH_PP_SCHEDULE" in res["schedule_bad"]
    # Default schedule: 1F1B, marked automatic (a step with fewer micro-batches than stages runs as GPipe), so
    # UNSLOTH_PP_MICROBATCHES below the stage count stays accepted; an explicit schedule is kept as given.
    assert res["pp2"][6] == "1f1b" and res["pp2"][8] is True, res["pp2"]
    assert res["schedule"][8] is False and res["schedule_gpipe"][6:9:2] == ("gpipe", False)
    # Fewer micro-batches than stages: the layout already records the GPipe fallback (sharded loading reads it).
    assert res["default_mb1"][5:7] == (1, "gpipe") and res["default_mb1"][8] is True, res[
        "default_mb1"
    ]
    assert "UNSLOTH_PP_MICROBATCHES >= UNSLOTH_PP_SIZE" in res["schedule_mb"], res["schedule_mb"]
    assert (
        "native FSDP2 only" in res["accelerate_fsdp"]
        and "UNSLOTH_PP_SIZE=2" in res["accelerate_fsdp"]
    )
    assert (
        "vLLM" in res["refuse_vllm"]
        and "UNSLOTH_PP_SIZE=2 (pipeline parallel)" in res["refuse_vllm"]
    )
    assert "full finetuning" in res["refuse_full_ft"] and res["refused_before_load"]
    for key in ("dpo", "grpo"):
        assert (
            "Trainer / SFTTrainer only" in (res[key] or "") and "UNSLOTH_PP_SIZE=2" in res[key]
        ), res[key]
    assert "overrides compute_loss" in (res["custom_loss"] or ""), res["custom_loss"]
    assert res["plain"] is None and res["trainer"] is None, (res["plain"], res["trainer"])
    assert "load_best_model_at_end" in (res["best"] or ""), res["best"]
    assert "label_smoothing_factor" in (res["smoothing"] or ""), res["smoothing"]
    assert "compute_loss_func" in (res["loss_func"] or ""), res["loss_func"]
    assert "optimizers = (optimizer" in (res["prebuilt_optimizer"] or ""), res["prebuilt_optimizer"]
    assert res["no_prebuilt_optimizer"] is None, res["no_prebuilt_optimizer"]
    # Failing before: parameter groups passed through optimizer_cls_and_kwargs went around the guard.
    assert "optimizers = (optimizer" in (res["prebuilt_param_groups"] or ""), res[
        "prebuilt_param_groups"
    ]
    if res["sft"] != "no trl":
        assert res["sft"] is None, res["sft"]
        assert "packing" in (res["packing"] or ""), res["packing"]
    # Since #18165 a mixture-of-experts model loads under pp with its router aux loss off (the default); the
    # aux-on refusal is covered in test_pp_moe_aux_off_gloo.py.
    assert res["moe"] is None, res["moe"]
    assert res["qwen3_moe"] is None, res["qwen3_moe"]
    assert res["act_dtype"] == (torch.float16, torch.float32), res["act_dtype"]
    assert res["stub_dtype"] == torch.float32, res["stub_dtype"]


@needs_pipelining
@pytest.mark.parametrize(
    "world,env,ref_bs",
    [
        (2, {"UNSLOTH_PP_SIZE": "2"}, 2),
        (4, {"UNSLOTH_PP_SIZE": "2", "UNSLOTH_TP_SIZE": "2"}, 2),
        (4, {"UNSLOTH_PP_SIZE": "2"}, 4),
        (4, {"UNSLOTH_PP_SIZE": "2", "UNSLOTH_FSDP2": "1"}, 4),
    ],
    ids = ["pp2", "pp2_x_tp2", "pp2_x_ddp2", "pp2_x_fsdp2_dp2"],
)
def test_pp_env_trainer_matches_one_process(world, env, ref_bs):
    ref = _ref(ref_bs, eval_bs = 2)  # eval batches as each dp replica's
    results = _spawn("env", world, env, bs = 2)
    stages = set()
    for rank, res in enumerate(results):
        assert res["is_pp"] and res["pre_split"], res  # whole until the Trainer
        if res["stage"] == 1:
            assert res["stage_ids"] and sum(res["stage_ids"]) == sum(res["seen"]), (
                res["stage_ids"],
                res["seen"],
            )
        assert "DistributedDataParallel" not in res["wrapped"]
        assert res["sampler_hooked"], res
        # GA 2: each window of 2 batches ran as one schedule step over all 4 rows.
        assert res["fold"]["windows"] == 3 and res["fold"]["schedule_steps"] == 3, res["fold"]
        _close(res, ref, what = rank)
        stages.add(res["stage"])
    assert stages == {0, 1}
    owned = [set(r["owned"]) for r in results if r["stage"] == 0][0] | [
        set(r["owned"]) for r in results if r["stage"] == 1
    ][0]
    assert not (
        [set(r["owned"]) for r in results if r["stage"] == 0][0]
        & [set(r["owned"]) for r in results if r["stage"] == 1][0]
    )
    assert owned, owned
    _same_tensors(results[0]["adapter"], ref["adapter"], what = "save_pretrained")
    _same_tensors(results[0]["trainer_adapter"], ref["adapter"], what = "Trainer.save_model")
    dp = 2 if world == 4 and "UNSLOTH_TP_SIZE" not in env else 1
    by_replica = {}
    for rank, res in enumerate(results):
        by_replica.setdefault(rank % dp if dp > 1 else 0, []).append(res["seen"])
    for replica, seen in by_replica.items():
        assert all(s == seen[0] for s in seen), (
            replica,
            seen,
        )  # every stage of a replica: same rows
    if dp > 1:
        assert by_replica[0][0] != by_replica[1][0]
    replica_ok = "UNSLOTH_TP_SIZE" not in env and "UNSLOTH_FSDP2" not in env
    for rank, res in enumerate(results):
        if replica_ok:
            assert res["kept_on_cpu"] is True, res["kept_on_cpu"]
            assert res["generated"] == ref["generated"], (rank, res["generated"], ref["generated"])
            assert res["replica_active"] and res["released"], res
            assert res["replica_training"] == [], res["replica_training"]
            assert res["released_by_save"], res
            assert res["after_generate"] is None, res["after_generate"]
            if rank == 0:
                _same_tensors(res["export"], ref["export"], what = "export")
            else:
                assert res["export_other"] == sorted(ref["export"]), res["export_other"]
        else:
            assert "not supported yet on a pipeline parallel x" in (
                res["generate_refused"] or ""
            ), res
            assert "UNSLOTH_PP_SIZE=2" in res["generate_refused"], res["generate_refused"]
            assert res["export_refused"] == res["generate_refused"].replace(
                "`generate`", "`export`"
            )


@needs_pipelining
@pytest.mark.parametrize(
    "ga,bs,mb,schedule,arch",
    [
        (1, 4, None, None, "llama"),
        (4, 2, None, None, "llama"),
        (2, 4, "1", None, "llama"),
        (2, 1, None, "1f1b", "llama"),
        (1, 4, None, None, "gemma3"),
        (2, 2, "2", "interleaved1f1b", "llama"),
    ],
    ids = [
        "ga1_bs4",
        "ga4_bs2",
        "mb1",
        "bs1_1f1b_padded",
        "gemma3_ga1_bs4",
        "interleaved1f1b_ga2_bs2_unfolded",
    ],
)
def test_pp_env_gradient_accumulation_and_microbatches(ga, bs, mb, schedule, arch):
    if mb == "1" and tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (
        2,
        13,
    ):
        # torch 2.10's gpipe needs micro-batches >= stages; Unsloth raises a ValueError naming both counts.
        pytest.skip("gpipe with 1 micro-batch over 2 stages needs torch >= 2.13")
    # bs1_1f1b_padded: 1F1B needs 2 micro-batches, so the 1-row batch gets a fully masked filler row.
    # gemma3_ga1_bs4: the Gemma3_(270M) notebook's shape (scaled tied embedding, sliding layers).
    # interleaved1f1b: one GA 2 case (stage layout, schedule, no fold, parity) over 2 steps keeps the gloo-pp shard short.
    steps = 2 if schedule == "interleaved1f1b" else 3
    ref = _ref(bs, ga = ga, steps = steps, evaluate = False, post = False, arch = arch)
    env = {"UNSLOTH_PP_SIZE": "2"}
    if mb:
        env["UNSLOTH_PP_MICROBATCHES"] = mb
    if schedule:
        env["UNSLOTH_PP_SCHEDULE"] = schedule
    if schedule == "interleaved1f1b":
        env["UNSLOTH_PP_KEEP_REPLICA"] = "0"  # the whole-model replica needs one stage per rank
    for rank, res in enumerate(
        _spawn("env", 2, env, bs = bs, ga = ga, steps = steps, evaluate = False, post = False, arch = arch)
    ):
        if schedule == "interleaved1f1b":
            # 4 layers over 2 ranks x 2 stages: rank r keeps stages r and r + 2
            assert res["local_stages"] == (rank, rank + 2), res["local_stages"]
            assert res["step_schedules"] == ["ScheduleInterleaved1F1B"], res["step_schedules"]
            assert not res["fold"].get("windows"), res["fold"]
        _close(res, ref, what = rank)
        # _close already pins every loss and grad norm. The trained adapter after several steps carries
        # more of the reduction-order noise: GitHub runners put gemma3_ga1_bs4's lora_A 1.6e-5 from the
        # reference on two PRs (1 of 64 elements), against at most 3.7e-6 here.
        _same_tensors(res["adapter"] or ref["adapter"], ref["adapter"], atol = 5e-5)


@pytest.fixture
def checkout_unsloth():
    """`import unsloth` from this checkout in the pytest process, then put the module table back.

    _import_checkout_unsloth drops every unsloth.* module first; left that way, later tests on the same worker
    lost the stub-loaded modules they read from sys.modules (KeyError: 'unsloth.models.pp_schedules')."""
    saved = {n: m for n, m in sys.modules.items() if n == "unsloth" or n.startswith("unsloth.")}
    _import_checkout_unsloth()
    yield
    for name in [n for n in sys.modules if n == "unsloth" or n.startswith("unsloth.")]:
        del sys.modules[name]
    sys.modules.update(saved)


def test_pp_default_schedule_plan(checkout_unsloth):
    from unsloth import parallel_env
    from unsloth.models import pp_env

    assert parallel_env.DEFAULT_PP_SCHEDULE == "1f1b"
    # Default 2 x pp micro-batches is unchanged: only the schedule default moved.
    stage = type(
        "S", (), {"env_layout": None, "n_microbatches": 4, "num_stages": 2, "schedule": "1f1b"}
    )()
    stage.schedule_auto = True
    assert pp_env.resolve_microbatches(stage, 8) == (4, 0)
    if pp_env._gpipe_needs_stage_count():
        return  # torch 2.10: GPipe needs >= stages micro-batches too, so both pad
    # A 1-row batch: the default 1F1B runs that step as GPipe (no filler row); an explicit 1F1B pads.
    assert pp_env._plan_microbatches(4, None, 2, "1f1b", 1, True, auto = True) == (1, 0)
    assert pp_env._plan_microbatches(4, None, 2, "1f1b", 1, True, auto = False) == (2, 1)


@needs_pipelining
@pytest.mark.parametrize(
    "bs,ga,schedule,expect",
    [
        (4, 1, None, ["Schedule1F1B"]),
        (4, 1, "gpipe", ["ScheduleGPipe"]),
        (2, 2, None, ["Schedule1F1B"]),
        (1, 1, None, ["ScheduleGPipe"]),
        # GA 2 folds the window's two 1-row batches into one step of 2 micro-batches: 1F1B again.
        (1, 2, None, ["Schedule1F1B"]),
    ],
    ids = [
        "default_bs4",
        "gpipe_bs4",
        "default_bs2_ga2",
        "default_bs1_falls_back",
        "default_bs1_ga2_folds",
    ],
)
def test_pp_env_default_schedule_same_loss(bs, ga, schedule, expect, checkout_unsloth):
    # The default schedule moved from GPipe to 1F1B: same micro-batches, same per-token loss normalisation, so
    # every logged loss, grad norm and the trained adapter equal the one-process run (and so each other).
    from unsloth.models import pp_env

    if bs < 2 and pp_env._gpipe_needs_stage_count():
        pytest.skip(
            "torch 2.10's GPipe needs micro-batches >= stages: a 1-row batch pads either way"
        )
    ref = _ref(bs, ga = ga, evaluate = False, post = False)
    env = {"UNSLOTH_PP_SIZE": "2"}
    if schedule:
        env["UNSLOTH_PP_SCHEDULE"] = schedule
    for rank, res in enumerate(_spawn("env", 2, env, bs = bs, ga = ga, evaluate = False, post = False)):
        assert res["schedule"] == ("1f1b" if schedule is None else schedule, schedule is None), res[
            "schedule"
        ]
        assert res["step_schedules"] == expect, res["step_schedules"]
        _close(res, ref, what = rank)
        _same_tensors(res["adapter"] or ref["adapter"], ref["adapter"], atol = 5e-5)


@needs_pipelining
def test_pp_env_stages_train_on_the_first_stage_batch():
    # Failing before: each stage collated its own batch, so with a stochastic collator the last stage scored labels
    # for tokens the first stage never embedded.
    results = _spawn(
        "env", 2, {"UNSLOTH_PP_SIZE": "2"}, bs = 2, noisy = True, evaluate = False, post = False
    )
    first, last = sorted(results, key = lambda r: r["stage"])
    assert first["seen"] != last["seen"]  # the collators did diverge
    assert last["stage_ids"] and sum(last["stage_ids"]) == sum(first["seen"]), (
        last["stage_ids"],
        first["seen"],
    )


@needs_pipelining
@pytest.mark.parametrize("world", [2, 4], ids = ["pp2", "pp2_x_ddp2"])
def test_pp_env_checkpoint_resume_equals_uninterrupted(world):
    for rank, res in enumerate(_spawn("resume", world, {"UNSLOTH_PP_SIZE": "2"})):
        a, b = res["a"], res["b"]
        assert a["steps"] == [1, 2, 3, 4] and b["steps"] == [1, 2, 3, 4], (a, b)
        assert (
            "adapter_model.safetensors" in res["files"] and f"rank{rank:05d}.pt" in res["files"]
        ), res["files"]
        assert b["losses"][2:] == a["losses"][2:] and b["norms"][2:] == a["norms"][2:], (rank, a, b)
        assert res["same"], rank


@needs_pipelining
def test_pp_env_fp16_scaler_unscales_once():
    for res in _spawn("scaler_clip", 2):
        assert (
            res["norm"] > 0 and res["max_rel"] < 1e-3
        ), res  # update == -lr x the unscaled gradient
    for res in _spawn("scaler_clip", 2, wrapper_unscale = True):
        assert (
            res["max_rel"] > 0.9
        ), res  # unscaled twice: the update is 1 / 1024 of it (the bug this pins)


@needs_pipelining
def test_pp_env_tensor_peers_train_on_one_batch():
    # Failing before: only the pipeline group took the first stage's batch, so under pp=2,tp=2 a stochastic
    # collator left tensor parallel peers [0, 1] / [2, 3] reducing activations of different examples.
    results = _spawn(
        "env",
        4,
        {"UNSLOTH_PP_SIZE": "2", "UNSLOTH_TP_SIZE": "2"},
        bs = 2,
        noisy = True,
        evaluate = False,
        post = False,
    )
    assert results[0]["seen"] != results[1]["seen"]  # the tensor peers' collators did diverge
    first = [r for r in results if r["stage"] == 0]
    last = [
        r for r in results if r["stage"] == 1
    ]  # the scoring stage records the ids it trained on
    sums = [sum(r["stage_ids"]) for r in last]
    assert len(last) == 2 and all(r["stage_ids"] for r in last), sums
    assert sums == [sum(first[0]["seen"])] * 2, (sums, [sum(r["seen"]) for r in first])


@needs_pipelining
@pytest.mark.parametrize(
    "world,env,bs,ga,ref_bs",
    [
        (2, {}, 2, 4, 2),
        (2, {}, 1, 8, 1),
        (2, {"UNSLOTH_PP_SCHEDULE": "1f1b"}, 1, 4, 1),
        (2, {"UNSLOTH_PP_SCHEDULE": "gpipe"}, 1, 4, 1),
        (2, {"UNSLOTH_PP_MICROBATCHES": "1"}, 2, 3, 2),
        (4, {}, 1, 4, 2),
        (4, {"UNSLOTH_FSDP2": "1"}, 1, 4, 2),
        (4, {"UNSLOTH_TP_SIZE": "2"}, 2, 4, 2),
    ],
    ids = [
        "bs2_ga4",
        "bs1_ga8",
        "bs1_ga4_1f1b",
        "bs1_ga4_gpipe",
        "bs2_ga3_mb1",
        "pp2_x_ddp2_bs1_ga4",
        "pp2_x_fsdp2_bs1_ga4",
        "pp2_x_tp2_bs2_ga4",
    ],
)
def test_pp_env_fold_ga_window_matches_one_process(world, env, bs, ga, ref_bs):
    # The accumulation window's rows (2 to 9 tokens, padded per batch) run as few schedule steps over all
    # of them, grouped by length: every logged loss, grad norm and the trained adapter equal one process
    # running the same batches with gradient accumulation; UNSLOTH_PP_FOLD_GA=0 runs one step per batch.
    if env.get("UNSLOTH_PP_MICROBATCHES") == "1" and tuple(
        int(x) for x in torch.__version__.split("+")[0].split(".")[:2]
    ) < (2, 13):
        pytest.skip("gpipe with 1 micro-batch over 2 stages needs torch >= 2.13")
    kw = dict(bs = bs, ga = ga, steps = 2, evaluate = False, post = False, varlen = True)
    ref = _ref(ref_bs, **{k: v for k, v in kw.items() if k != "bs"})
    base = {"UNSLOTH_PP_SIZE": "2", **env}
    for fold in (True, False):
        run_env = base if fold else {**base, "UNSLOTH_PP_FOLD_GA": "0"}
        for rank, res in enumerate(_spawn("env", world, run_env, **kw)):
            stats = res["fold"]
            if fold:
                # Engaged: 2 windows, fewer schedule steps than batches.
                assert stats["windows"] == 2 and stats["batches"] == 2 * ga, stats
                assert stats["schedule_steps"] < stats["batches"], stats
            else:
                assert stats == {}, stats
            _close(res, ref, what = (fold, rank))
            _same_tensors(res["adapter"] or ref["adapter"], ref["adapter"], atol = 5e-5)


def test_pp_fold_plan(checkout_unsloth):
    from unsloth.models import pp_env

    plan = pp_env._fold_plan
    # Equal rows: one schedule step over the window (bubble once, not once per batch).
    assert plan([100] * 8, 1, 2, 1, None) == [(0, 8, 8)]
    # One long row and short ones: the long row alone, the rest together (no padding to the long one).
    assert plan([1000] + [50] * 7, 1, 2, 1, None) == [(0, 1, 1), (1, 8, 7)]
    # Explicit 1F1B needs >= stages micro-batches per step (the long row takes a real neighbour, not a
    # filler); GPipe keeps at most max(m, stages) in flight.
    assert plan([1000] + [50] * 7, 1, 2, 2, None) == [(0, 2, 2), (2, 8, 6)]
    assert plan([100] * 8, 1, 2, 1, 2) == [(0, 2, 2), (2, 4, 2), (4, 6, 2), (6, 8, 2)]
    # Whole micro-batches of 2 rows; the last one is filled up.
    assert plan([100] * 7, 2, 2, 1, None) == [(0, 7, 4)]


@needs_pipelining
def test_pp_fold_needs_every_dp_replica_to_agree():
    # Failing before: each replica planned its window from its own first batch, so replicas with different
    # rows per batch could run different numbers of schedule steps and hang in the dp collectives.
    for res in _spawn("dp_agree", 2):
        assert res == {"same": True, "one_off": False, "rows_differ": False}, res


@needs_pipelining
def test_pp_fold_window_with_different_side_inputs_runs_per_batch():
    # Failing before: a window whose batches carried different side inputs (attention_mask on some only)
    # was stacked with the first batch's keys (KeyError, or a mask dropped). It now runs one step per batch.
    kw = dict(bs = 2, ga = 4, steps = 2, evaluate = False, post = False, varlen = True, mixed_keys = True)
    ref = _ref(2, **{k: v for k, v in kw.items() if k != "bs"})
    for rank, res in enumerate(_spawn("env", 2, {"UNSLOTH_PP_SIZE": "2"}, **kw)):
        assert res["fold"] == {}, res["fold"]
        _close(res, ref, what = rank)


@needs_pipelining
def test_pp_fold_keeps_length_dependent_rope_per_batch():
    # Failing before: rows regrouped and re-padded across the window changed max(position_ids), so dynamic NTK
    # RoPE (HF dynamic_rope_update) rotated with other frequencies than one process. Such a model runs one
    # schedule step per batch.
    kw = dict(bs = 2, ga = 4, steps = 2, evaluate = False, post = False, varlen = True)
    kw["arch"] = "llama_dynamic_rope"
    ref = _ref(2, **{k: v for k, v in kw.items() if k != "bs"})
    for rank, res in enumerate(_spawn("env", 2, {"UNSLOTH_PP_SIZE": "2"}, **kw)):
        _close(res, ref, what = rank)  # folded: step 1 grad norm 0.79173 vs 0.79219
        assert res["fold"] == {}, res["fold"]


def test_pp_pack_plan(checkout_unsloth):
    from unsloth.models import pp_env

    # Equal rows pack as they are; the length groups cost the same, so packing is not chosen there.
    assert pp_env._pack_plan([100] * 8, 1, 2, 1, None)[:3] == (900, 8, 100)
    lengths = [510, 300, 250, 200, 180, 150, 100, 60]
    cost, m, width, where = pp_env._pack_plan(lengths, 1, 2, 1, None)
    groups = pp_env._fold_cost(pp_env._fold_plan(lengths, 1, 2, 1, None), lengths, 1, 2)
    # Whole documents in rows no longer than the longest one (what the unfolded step held).
    assert (cost, m, width) == (2550, 4, 510) and cost < groups == 3020
    load = [sum(x for x, b in zip(lengths, where) if b == i) for i in range(m)]
    assert max(load) <= width and sorted(set(where)) == list(range(m)), load
    # Micro-batches of 2 rows: the length groups' cost counts both rows (token work), as the packed rows do.
    twice = sorted(lengths * 2, reverse = True)
    groups2 = pp_env._fold_cost(pp_env._fold_plan(twice, 2, 2, 1, None), twice, 2, 2)
    assert groups2 == 6040 and pp_env._pack_plan(twice, 2, 2, 1, None)[0] < groups2
    # GPipe's in-flight cap bounds the micro-batch count: 41 tokens do not fit 2 rows of at most 9.
    assert pp_env._pack_plan([9, 8, 8, 5, 4, 3, 2, 2], 1, 2, 1, 2) is None
    assert pp_env._pack_plan([9, 8, 8, 5, 4, 3, 2, 2], 3, 2, 1, 2)[1:3] == (2, 27)
    # Large windows plan quickly: lower-bounded counts are skipped, the width is bisected (seconds before).
    import random
    import time

    rng = random.Random(0)
    many = sorted((rng.randint(32, 1024) for _ in range(1024)), reverse = True)
    start = time.perf_counter()
    cost, m, width, where = pp_env._pack_plan(many, 2, 2, 1, None)
    assert time.perf_counter() - start < 2.0
    assert (
        len(where) == 1024
        and max(sum(x for x, b in zip(many, where) if b == i) for i in range(m)) <= width
    )
    # Packed rows: documents back to back, positions restart per document and for the padding tail, the
    # first label of every document ignored.
    rows = [
        (3, torch.tensor([[1, 2, 3]]), torch.tensor([[1, 2, 3]]), {}),
        (2, torch.tensor([[4, 5]]), torch.tensor([[4, 5]]), {}),
        (1, torch.tensor([[6]]), torch.tensor([[6]]), {}),
    ]
    ids, labels, extra = pp_env._stack_packed(rows, 2, 4, [0, 1, 1], 2)
    assert ids.tolist() == [[1, 2, 3, 0], [4, 5, 6, 0]]
    assert labels.tolist() == [[-100, 2, 3, -100], [-100, 5, -100, -100]]
    assert extra["position_ids"].tolist() == [[0, 1, 2, 0], [0, 1, 0, 0]]
    # Left padding / non-default positions are not one plain document per row.
    masked = (2, rows[1][1], rows[1][2], {"attention_mask": torch.tensor([[0, 1]])})
    shifted = (2, rows[1][1], rows[1][2], {"position_ids": torch.tensor([[3, 4]])})
    assert (
        pp_env._plain_rows(rows)
        and not pp_env._plain_rows([masked])
        and not pp_env._plain_rows([shifted])
    )


@needs_pipelining
@pytest.mark.parametrize(
    "world,env,bs,ga,ref_bs",
    [
        (2, {}, 1, 8, 1),
        (2, {}, 2, 4, 2),
        (2, {"UNSLOTH_PP_SCHEDULE": "1f1b"}, 1, 4, 1),
        (4, {}, 1, 4, 2),
        (4, {"UNSLOTH_FSDP2": "1"}, 1, 4, 2),
    ],
    ids = ["bs1_ga8", "bs2_ga4", "bs1_ga4_1f1b", "pp2_x_ddp2_bs1_ga4", "pp2_x_fsdp2_bs1_ga4"],
)
def test_pp_fold_packs_documents_matches_one_process(world, env, bs, ga, ref_bs):
    # Failing before: each window's rows ran in length groups padded to their longest. Now the window's
    # documents are packed back to back into equal-length rows (positions restarting per document, so the
    # attention keeps them apart) whenever that is cheaper: every logged loss, grad norm and the adapter still
    # equal one process. UNSLOTH_PP_PACK=0 keeps the length groups.
    kw = dict(bs = bs, ga = ga, steps = 2, evaluate = False, post = False, varlen = True)
    ref = _ref(ref_bs, **{k: v for k, v in kw.items() if k != "bs"})
    base = {"UNSLOTH_PP_SIZE": "2", **env}
    for pack in (True, False):
        run_env = base if pack else {**base, "UNSLOTH_PP_PACK": "0"}
        for rank, res in enumerate(_spawn("env", world, run_env, **kw)):
            stats = res["fold"]
            assert stats["windows"] == 2 and stats["batches"] == 2 * ga, stats
            if pack:
                # Engaged where cheaper than the length groups (each packed window is one schedule step).
                assert stats.get("packed", 0) >= 1, stats
            else:
                assert stats.get("packed") == 0 and stats["schedule_steps"] >= 2, stats
            _close(res, ref, what = (pack, rank))
            _same_tensors(res["adapter"] or ref["adapter"], ref["adapter"], atol = 5e-5)


@needs_pipelining
def test_pp_fold_does_not_pack_length_dependent_rope():
    # Dynamic NTK RoPE keeps one schedule step per batch: no fold, so no packing either.
    kw = dict(bs = 2, ga = 4, steps = 2, evaluate = False, post = False, varlen = True)
    for rank, res in enumerate(
        _spawn("env", 2, {"UNSLOTH_PP_SIZE": "2"}, arch = "llama_dynamic_rope", **kw)
    ):
        assert res["fold"] == {}, res["fold"]


def test_pp_fold_filler_row_is_attended(checkout_unsloth):
    # Failing before: a dp replica that ran out of rows got a filler copied from its first row's first
    # column, which is masked out when the rows are left padded. A fully masked row gives NaN attention.
    from unsloth.models import pp_env

    ids = torch.tensor([[0, 0, 7, 8]])
    rows = [
        (
            4,
            ids,
            ids.clone(),
            {"attention_mask": torch.tensor([[0, 0, 1, 1]]), "position_ids": ids * 0},
        )
    ]
    input_ids, labels, extra = pp_env._stack_group(rows, 1, 1, 2, 1)
    assert extra["attention_mask"].tolist() == [[1], [1]], extra
    assert labels.tolist() == [[-100], [-100]] and input_ids.shape == (2, 1)


def test_pp_fold_shares_the_token_budget_exactness_guards(checkout_unsloth):
    # Failing before: folding regrouped and re-padded rows under NEFTune (noise scaled by each forward's
    # length) and LOMO / layerwise optimizers (weights updated on every backward). Both now keep one
    # schedule step per batch, through the token budget's own guard; RoPE kinds are case-folded.
    import types

    from unsloth.models import pp_env
    from unsloth.utils.token_budget import _exactness_blocker

    def trainer(**args):
        base = dict(optim = "adamw_torch", neftune_noise_alpha = None)
        return types.SimpleNamespace(args = types.SimpleNamespace(**{**base, **args}), model = None)

    assert _exactness_blocker(trainer(), step_loss = False) is None
    assert "NEFTune" in _exactness_blocker(trainer(neftune_noise_alpha = 5.0), step_loss = False)
    assert "backward" in _exactness_blocker(trainer(optim = "lomo"), step_loss = False)
    assert "backward" in _exactness_blocker(
        trainer(optim = "galore_adamw_layerwise"), step_loss = False
    )
    assert pp_env._rope_types({"rope_scaling": {"rope_type": "Dynamic"}}) == {"dynamic"}


@needs_pipelining
def test_pp_fold_skips_neftune():
    # Failing before: NEFTune training folded the window (its noise follows each forward's padded length).
    kw = dict(bs = 2, ga = 4, steps = 2, evaluate = False, post = False, varlen = True, neftune = 5.0)
    for res in _spawn("env", 2, {"UNSLOTH_PP_SIZE": "2"}, **kw):
        assert res["fold"] == {}, res["fold"]
