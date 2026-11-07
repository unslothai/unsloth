# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A context parallel SFTTrainer binds its wrappers on itself only: a plain SFTTrainer built later in the same
process (same gloo ranks) runs the class's original methods, and no CP trainer wrapper is called during its
steps. With class-level wrappers every later trainer paid for them (about 35% of a torch 2.10 LoRA step with CP
off); here that is settled by call counts, not timing."""

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
from test_parallel_cp_compose_gloo import _ENV_KEYS, SEQ, _checkpoint  # noqa: E402
from test_tp_native_gloo import _free_port, _import_checkout_unsloth  # noqa: E402

# The trainer-level CP wrappers (the attention / data hooks are per model and not counted here).
_WRAPPERS = {
    "patched_training_step",
    "patched_prediction_step",
    "patched_train",
    "patched_evaluate",
    "patched_predict",
    "patched_evaluation_loop",
    "patched_save_model",
    "_save_model",
    "patched_get_total_train_batch_size",
    "_install_accelerator_state",
    "_cp_state_scope",
    "call",  # the old pre-CP trainers' restore wrapper
}
_METHODS = (
    "training_step",
    "prediction_step",
    "train",
    "evaluate",
    "predict",
    "evaluation_loop",
    "save_model",
    "get_total_train_batch_size",
)


def _case(rank, world, tmp, ckpt):
    error = _import_checkout_unsloth()
    if error is not None:
        return {"skip": error}
    import datasets
    import torch.distributed as dist
    import transformers
    import trl
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from unsloth import context_parallel as cpm
    from unsloth.models._utils import patch_gradient_accumulation_fix

    patch_gradient_accumulation_fix(transformers.Trainer)
    cpm.patch_sft_trainer()
    cpm.DEVICE_TYPE_TORCH = "cpu"
    from unsloth.utils import attention_dispatch

    attention_dispatch.HAS_XFORMERS = False
    attention_dispatch.HAS_FLASH_ATTENTION = False
    cls = trl.SFTTrainer
    class_before = {name: cls.__dict__.get(name) for name in _METHODS}
    resolved_before = {name: getattr(cls, name, None) for name in _METHODS}

    def model():
        m = AutoModelForCausalLM.from_pretrained(
            ckpt, torch_dtype = torch.float32, attn_implementation = "sdpa"
        )
        torch.manual_seed(1)
        return get_peft_model(
            m, LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", lora_dropout = 0.0)
        )

    g = torch.Generator().manual_seed(1)
    data = datasets.Dataset.from_dict(
        {"input_ids": torch.randint(0, 1000, (16, SEQ), generator = g).tolist()}
    )

    def collate(rows):
        ids = torch.tensor([r["input_ids"] for r in rows])
        return {"input_ids": ids, "labels": ids.clone()}

    def config(name, cp):
        args = trl.SFTConfig(
            output_dir = os.path.join(tmp, name),
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 2,
            max_steps = 2,
            learning_rate = 1e-3,
            optim = "adamw_torch",
            logging_steps = 1,
            save_strategy = "no",
            report_to = [],
            use_cpu = True,
            seed = 0,
            remove_unused_columns = False,
            disable_tqdm = True,
            dataset_kwargs = {"skip_prepare_dataset": True},
        )
        if cp > 1:
            args.context_parallel_size = cp
            args.context_parallel_backend = "ulysses"
        return args

    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    calls = {}

    def profile(frame, event, arg):
        if (
            event == "call"
            and frame.f_code.co_filename == cpm.__file__
            and frame.f_code.co_name in _WRAPPERS
        ):
            calls[frame.f_code.co_name] = calls.get(frame.f_code.co_name, 0) + 1

    def run(trainer):
        calls.clear()
        sys.setprofile(profile)
        try:
            trainer.train()
        finally:
            sys.setprofile(None)
        return dict(calls)

    cp_model = model()  # Hugging Face attention: the trainer installs generic Ulysses itself
    cp_trainer = cls(
        model = cp_model,
        args = config("cp", 2),
        train_dataset = data,
        data_collator = collate,
        processing_class = tokenizer,
    )
    assert cp_trainer._context_parallel_manager is not None
    cp_calls = run(cp_trainer)

    plain = cls(
        model = model(),
        args = config("plain", 1),
        train_dataset = data,
        data_collator = collate,
        processing_class = tokenizer,
    )
    plain_calls = run(plain)
    # What a plain trainer resolves each method to, and whether a CP wrapper sits behind it.
    plain_wrapped = []
    for name in _METHODS:
        method = getattr(plain, name, None)
        chain, f = [], getattr(method, "__func__", method)
        while f is not None and len(chain) < 20:
            chain.append(f)
            f = getattr(
                getattr(f, "__wrapped__", None), "__func__", getattr(f, "__wrapped__", None)
            )
        if any(
            getattr(getattr(c, "__code__", None), "co_filename", None) == cpm.__file__
            for c in chain
        ):
            plain_wrapped.append(name)
    res = {
        "cp_calls": cp_calls,
        "plain_calls": plain_calls,
        "plain_wrapped": plain_wrapped,
        "class_unchanged": [n for n in _METHODS if cls.__dict__.get(n) is class_before[n]],
        "resolved_unchanged": [n for n in _METHODS if getattr(cls, n, None) is resolved_before[n]],
        "plain_losses": [h["loss"] for h in plain.state.log_history if "loss" in h],
    }
    if dist.is_initialized():
        dist.barrier()
    return res


def _worker(rank, world, port, tmp, ckpt):
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
    torch.set_num_threads(1)
    import torch.distributed as dist

    try:
        try:
            res = _case(rank, world, tmp, ckpt)
        except Exception as e:
            import traceback
            res = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


def test_a_plain_trainer_after_a_cp_trainer_runs_the_original_methods():
    ctx = multiprocessing.get_context("spawn")
    port, world = _free_port(), 2
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _checkpoint(tmp)
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, tmp, ckpt)) for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(900)
            if p.is_alive():
                p.kill()
        results = []
        for r in range(world):
            path = os.path.join(tmp, f"result_{r}.pt")
            results.append(
                torch.load(path, weights_only = False)
                if os.path.exists(path)
                else {"error": f"rank {r}"}
            )
    if "skip" in results[0]:
        pytest.skip(results[0]["skip"])
    for rank, res in enumerate(results):
        assert "error" not in res, (rank, res.get("error"), res.get("trace"))
        print(
            f"rank {rank}: cp trainer wrapper calls {res['cp_calls']}, plain trainer {res['plain_calls']}"
        )
        # The CP trainer engaged its wrappers (the counter can see them) ...
        assert (
            res["cp_calls"].get("patched_training_step", 0) >= 4
            and res["cp_calls"].get("patched_train") == 1
        )
        # ... the plain trainer built after it called none, and resolves to the class's original methods.
        assert res["plain_calls"] == {}, res["plain_calls"]
        assert res["plain_wrapped"] == [], res["plain_wrapped"]
        assert res["class_unchanged"] == list(_METHODS)
        assert res["resolved_unchanged"] == list(_METHODS)
        assert len(res["plain_losses"]) == 2
