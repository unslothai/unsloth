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
    "UNSLOTH_FSDP2",
    "UNSLOTH_FSDP2_BACKEND",
    "ACCELERATE_USE_FSDP",
)
N_LAYERS = 4


def _checkpoint(tmp, arch = "llama"):
    return _save_checkpoint(arch, os.path.join(tmp, "ckpt"), {"num_hidden_layers": N_LAYERS})


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
        return None if got is None else (got.pp, got.dp, got.tp, got.world, got.fsdp2, got.microbatches, got.schedule, repr(got))

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
    out["schedule_mb"] = layout(
        UNSLOTH_PP_SIZE = 4, UNSLOTH_PP_SCHEDULE = "1f1b", UNSLOTH_PP_MICROBATCHES = 2, WORLD_SIZE = 4
    )
    out["accelerate_fsdp"] = layout(
        UNSLOTH_PP_SIZE = 2, UNSLOTH_FSDP2 = 1, UNSLOTH_FSDP2_BACKEND = "accelerate", WORLD_SIZE = 4
    )

    calls = []

    def loader(model_name, load_in_4bit = True, fast_inference = False, full_finetuning = False, **kw):
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
    if SFTTrainer is not None:
        out["packing"] = _raised(lambda: pp_env.pp_trainer_refusals(SFTTrainer, packed, lay))
        out["sft"] = _raised(lambda: pp_env.pp_trainer_refusals(SFTTrainer, plain_sft, lay))
    else:
        out["packing"] = out["sft"] = "no trl"
    # A model the split refuses is refused at from_pretrained, naming the variable.
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-Qwen3MoeForCausalLM")
    moe = tp_env.tensor_parallel_env_load(
        lambda name, load_in_4bit = True: AutoModelForCausalLM.from_config(config)
    )
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(_free_port()), RANK = "0", LOCAL_RANK = "0")
    layout(UNSLOTH_PP_SIZE = 1, WORLD_SIZE = 1)  # (no world for the mesh here: refusal path only)
    os.environ["UNSLOTH_PP_SIZE"] = "2"
    tp_env._reset_env_layout()
    tp_env._LAYOUT = tp_env.TPEnvLayout(1, 1, 2, False, 2)
    tp_env._LAYOUT._mesh = object()  # the refusal comes from the model, not the mesh
    out["moe"] = _raised(lambda: moe("m", load_in_4bit = False))
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
        seen.append(batch["input_ids"].sum().item())
        return batch

    def load(model_name, load_in_4bit = True, **kwargs):  # FastLanguageModel.from_pretrained's flags
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
    )
    trainer = SequentialTrainer(model = model, args = args, train_dataset = data, data_collator = collate)
    trainer.train()
    logs = [h for h in trainer.state.log_history if "loss" in h]
    res = {
        "steps": [h["step"] for h in logs],
        "losses": [h["loss"] for h in logs],
        "grad_norms": [h.get("grad_norm") for h in logs],
        "seen": list(seen),
        "pre_split": pre_split,
        "layout": None if layout is None else repr(layout),
        "is_pp": pp_native.is_pipeline_parallel(trainer.model),
        "wrapped": type(trainer.model_wrapped).__name__,
    }
    stage = pp_native.pipeline_stage_of(trainer.model)
    if stage is not None:
        res["stage"] = stage.stage_index
        res["owned"] = sorted(n for n, p in trainer.model.named_parameters() if p.requires_grad)
        res["kept_on_cpu"] = all(
            p.device.type == "cpu" for _, _, m, _ in stage._kept for p in m.parameters()
        ) if stage._kept and stage.tp_mesh is None else None
    if evaluate:
        res["eval_loss"] = trainer.evaluate(eval_data)["eval_loss"]
    save_dir = os.path.join(tmp, "adapter")
    if stage is not None:
        pp_env.pp_save_adapter(trainer.model, (save_dir,), {})
    elif rank == 0:
        trainer.model.save_pretrained(save_dir)
    if dist.is_initialized():
        dist.barrier()
    res["adapter"] = load_file(os.path.join(save_dir, "adapter_model.safetensors")) if rank == 0 else None
    trainer.save_model(os.path.join(tmp, "trainer_save"))
    if dist.is_initialized():
        dist.barrier()
    res["trainer_adapter"] = (
        load_file(os.path.join(tmp, "trainer_save", "adapter_model.safetensors")) if rank == 0 else None
    )
    if post:
        prompt = ids[:1, :4]
        refusal = pp_env.replica_refusal(stage, "generate") if stage is not None else None
        if refusal is None:
            out = trainer.model.generate(input_ids = prompt, max_new_tokens = 4, do_sample = False, pad_token_id = 0)
            res["generated"] = out.tolist()
            if stage is not None:
                res["replica_active"] = stage.replica is not None
                # An export: the whole one-GPU state dict, built on rank 0 only (the others wait).
                full = pp_env.pp_collective(
                    trainer.model,
                    "export",
                    lambda: {k: v.detach().clone() for k, v in trainer.model.state_dict().items()},
                )
                res["export"] = full if rank == 0 else None
                # The other ranks return rank 0's result (as tp_replica's collective saves do).
                res["export_other"] = sorted(full) if rank != 0 else None
                pp_env.release_replica(trainer.model)
                res["released"] = stage.replica is None and all(
                    not isinstance(layer, torch.nn.Module) or type(layer).__name__ == "_PPPassThrough"
                    for i, layer in enumerate(stage.parts.layers)
                    if not (stage.start <= i < stage.end)
                )
                # A training step after generate: the pipeline split is back.
                trainer.args.max_steps = steps + 1
                res["after_generate"] = _raised(lambda: trainer.train(resume_from_checkpoint = None))
            else:
                res["export"] = {k: v.detach().clone() for k, v in trainer.model.state_dict().items()}
        else:
            res["generate_refused"] = _raised(
                lambda: trainer.model.generate(input_ids = prompt, max_new_tokens = 4, do_sample = False, pad_token_id = 0)
            )
            res["export_refused"] = _raised(lambda: pp_env.pp_collective(trainer.model, "export", lambda: None))
    if dist.is_initialized():
        dist.barrier()
    return res


CASES = {"units": _case_units, "env": _case_env}


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

    faulthandler.dump_traceback_later(max(int(os.environ.get("PP_TEST_TIMEOUT", 600)) - 30, 30), exit = True)
    import torch.distributed as dist

    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn(case, world = 1, env = None, **kwargs):
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        if case == "env":
            kwargs["ckpt"] = _checkpoint(tmp, kwargs.pop("arch", "llama"))
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, env or {}, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(int(os.environ.get('PP_TEST_TIMEOUT', 600)))
        codes = [p.exitcode for p in procs]
        for p in procs:
            if p.is_alive():
                p.kill()
        assert codes == [0] * world, f"rank exit codes {codes}"
        results = [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False) for r in range(world)
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
    not _pipelining_available(), reason = "torch.distributed.pipelining with step(return_outputs) (torch >= 2.10)"
)


def _close(got, want, rel = 1e-5, what = ""):
    assert got["steps"] == want["steps"], (what, got["steps"], want["steps"])
    assert got["losses"] == pytest.approx(want["losses"], rel = rel), (what, got["losses"], want["losses"])
    # Step 1 is tight; later steps carry the one-process reference's own run-to-run spread on this host
    # (3e-5 absolute at a 2.3 grad norm between two identical reference runs).
    assert got["grad_norms"][:1] == pytest.approx(want["grad_norms"][:1], rel = rel), (what, got["grad_norms"])
    assert got["grad_norms"] == pytest.approx(want["grad_norms"], rel = 1e-4), (
        what,
        got["grad_norms"],
        want["grad_norms"],
    )
    if "eval_loss" in want:
        assert got["eval_loss"] == pytest.approx(want["eval_loss"], rel = rel), (what, got["eval_loss"], want["eval_loss"])


def _same_tensors(got, want, rel = 1e-4, atol = 1e-5, what = ""):
    assert set(got) == set(want) and want, (what, sorted(set(got) ^ set(want))[:6])
    for key, value in want.items():
        torch.testing.assert_close(got[key], value, rtol = rel, atol = atol, msg = lambda m: f"{what} {key}: {m}")


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
    assert "UNSLOTH_PP_SIZE=3" in res["no_divide"] and "WORLD_SIZE=4" in res["no_divide"], res["no_divide"]
    assert "UNSLOTH_PP_SIZE x UNSLOTH_TP_SIZE = 4" in res["no_divide_tp"], res["no_divide_tp"]
    assert "must be >= 1" in res["zero"] and "must be an integer" in res["text"]
    assert res["mb"][5] == 3 and "UNSLOTH_PP_MICROBATCHES" in res["mb_text"]
    assert res["schedule"][6] == "1f1b" and "UNSLOTH_PP_SCHEDULE" in res["schedule_bad"]
    assert "UNSLOTH_PP_MICROBATCHES >= UNSLOTH_PP_SIZE" in res["schedule_mb"], res["schedule_mb"]
    assert "native FSDP2 only" in res["accelerate_fsdp"] and "UNSLOTH_PP_SIZE=2" in res["accelerate_fsdp"]
    assert "vLLM" in res["refuse_vllm"] and "UNSLOTH_PP_SIZE=2 (pipeline parallel)" in res["refuse_vllm"]
    assert "full finetuning" in res["refuse_full_ft"] and res["refused_before_load"]
    for key in ("dpo", "grpo"):
        assert "Trainer / SFTTrainer only" in (res[key] or "") and "UNSLOTH_PP_SIZE=2" in res[key], res[key]
    assert "overrides compute_loss" in (res["custom_loss"] or ""), res["custom_loss"]
    assert res["plain"] is None and res["trainer"] is None, (res["plain"], res["trainer"])
    assert "load_best_model_at_end" in (res["best"] or ""), res["best"]
    if res["sft"] != "no trl":
        assert res["sft"] is None, res["sft"]
        assert "packing" in (res["packing"] or ""), res["packing"]
    assert "mixture-of-experts" in (res["moe"] or "") and "UNSLOTH_PP_SIZE=2: unset it" in res["moe"], res["moe"]


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
        assert "DistributedDataParallel" not in res["wrapped"]
        _close(res, ref, what = rank)
        stages.add(res["stage"])
    assert stages == {0, 1}
    owned = [set(r["owned"]) for r in results if r["stage"] == 0][0] | [set(r["owned"]) for r in results if r["stage"] == 1][0]
    assert not ([set(r["owned"]) for r in results if r["stage"] == 0][0] & [set(r["owned"]) for r in results if r["stage"] == 1][0])
    assert owned, owned
    _same_tensors(results[0]["adapter"], ref["adapter"], what = "save_pretrained")
    _same_tensors(results[0]["trainer_adapter"], ref["adapter"], what = "Trainer.save_model")
    dp = 2 if world == 4 and "UNSLOTH_TP_SIZE" not in env else 1
    by_replica = {}
    for rank, res in enumerate(results):
        by_replica.setdefault(rank % dp if dp > 1 else 0, []).append(res["seen"])
    for replica, seen in by_replica.items():
        assert all(s == seen[0] for s in seen), (replica, seen)  # every stage of a replica: same rows
    if dp > 1:
        assert by_replica[0][0] != by_replica[1][0]
    replica_ok = "UNSLOTH_TP_SIZE" not in env and "UNSLOTH_FSDP2" not in env
    for rank, res in enumerate(results):
        if replica_ok:
            assert res["kept_on_cpu"] is True, res["kept_on_cpu"]
            assert res["generated"] == ref["generated"], (rank, res["generated"], ref["generated"])
            assert res["replica_active"] and res["released"], res
            assert res["after_generate"] is None, res["after_generate"]
            if rank == 0:
                _same_tensors(res["export"], ref["export"], what = "export")
            else:
                assert res["export_other"] == sorted(ref["export"]), res["export_other"]
        else:
            assert "not supported yet on a pipeline parallel x" in (res["generate_refused"] or ""), res
            assert "UNSLOTH_PP_SIZE=2" in res["generate_refused"], res["generate_refused"]
            assert res["export_refused"] == res["generate_refused"].replace("`generate`", "`export`")


@needs_pipelining
@pytest.mark.parametrize("ga,bs,mb", [(1, 4, None), (4, 2, None), (2, 4, "1")], ids = ["ga1_bs4", "ga4_bs2", "mb1"])
def test_pp_env_gradient_accumulation_and_microbatches(ga, bs, mb):
    ref = _ref(bs, ga = ga, evaluate = False, post = False)
    env = {"UNSLOTH_PP_SIZE": "2"}
    if mb:
        env["UNSLOTH_PP_MICROBATCHES"] = mb
    for rank, res in enumerate(_spawn("env", 2, env, bs = bs, ga = ga, evaluate = False, post = False)):
        _close(res, ref, what = rank)
        _same_tensors(res["adapter"] or ref["adapter"], ref["adapter"])
