# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""One gloo rank for test_plain_trainer_ddp_gc.py. No unsloth import at module level: a spawned rank must set up the GPU-free harness first."""

from __future__ import annotations

import os
import runpy
import tempfile


def run_rank(rank, port, cases, queue):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        LOCAL_RANK = str(rank),
        WORLD_SIZE = "2",
        LOCAL_WORLD_SIZE = "2",
        ACCELERATE_USE_CPU = "true",
    )
    runpy.run_path(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conftest.py")
    )
    import torch

    torch.set_num_threads(1)
    import unsloth  # noqa: F401
    import peft
    import transformers
    from unsloth.models._utils import patch_gradient_accumulation_fix

    # What FastLanguageModel / FastModel.from_pretrained installs on transformers.Trainer.
    patch_gradient_accumulation_fix(transformers.Trainer)

    out = {}
    try:
        for name, trainer_name, user_value in cases:
            torch.manual_seed(0)
            config = transformers.LlamaConfig(
                vocab_size = 64,
                hidden_size = 16,
                intermediate_size = 32,
                num_hidden_layers = 2,
                num_attention_heads = 2,
                num_key_value_heads = 1,
            )
            model = transformers.LlamaForCausalLM(config)
            # Reentrant checkpointing armed on the model (Unsloth's get_peft_model does this), not in the args.
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs = {"use_reentrant": True}
            )
            model.enable_input_require_grads()
            model = peft.get_peft_model(
                model,
                peft.LoraConfig(
                    r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj", "gate_proj"]
                ),
            )
            g = torch.Generator().manual_seed(7)
            rows = []
            for _ in range(8):
                ids = torch.randint(1, 64, (12,), generator = g)
                rows.append(
                    {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}
                )
            extra = {} if user_value is None else {"ddp_find_unused_parameters": user_value}
            args_cls = (
                transformers.Seq2SeqTrainingArguments
                if trainer_name == "Seq2SeqTrainer"
                else transformers.TrainingArguments
            )
            args = args_cls(
                output_dir = tempfile.mkdtemp(prefix = "ddp_gc_"),
                per_device_train_batch_size = 2,
                max_steps = 2,
                learning_rate = 1e-2,
                use_cpu = True,
                report_to = "none",
                save_strategy = "no",
                logging_steps = 1,
                dataloader_num_workers = 0,
                **extra,
            )
            trainer = getattr(transformers, trainer_name)(
                model = model, args = args, train_dataset = rows
            )
            error = None
            try:
                trainer.train()
            except RuntimeError as e:
                error = str(e).splitlines()[0] or type(e).__name__
            ddp = trainer.model_wrapped
            out[name] = dict(
                error = error,
                find_unused = getattr(ddp, "find_unused_parameters", None),
                is_ddp = isinstance(ddp, torch.nn.parallel.DistributedDataParallel),
                args_value = args.ddp_find_unused_parameters,
                losses = [h["loss"] for h in trainer.state.log_history if "loss" in h],
            )
            if error is not None:
                # A rank that raised leaves the group unusable for the next case.
                break
        queue.put((rank, out))
    except BaseException:
        import traceback
        queue.put((rank, {"error": traceback.format_exc()}))
        raise
    if not any(case.get("error") for case in out.values()):
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
