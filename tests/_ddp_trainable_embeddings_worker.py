# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""One DDP rank (or the single-process reference): LoRA + trainable embed_tokens / lm_head, reentrant
checkpointing. Each rank takes half the global batch so averaged gradients equal the reference."""

import importlib.util
import json
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_ddp_find_unused_default import _args, _llama, _lora  # noqa: E402

HELPER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth",
    "models",
    "_ddp_defaults.py",
)
STEPS = 3
GLOBAL_BATCH = 4


def _helper():
    spec = importlib.util.spec_from_file_location("_unsloth_ddp_defaults_embeddings", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.unsloth_default_ddp_find_unused_parameters


def main(out_dir):
    torch.set_num_threads(1)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("gloo")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1

    torch.manual_seed(0)
    model = _lora(
        _llama(),
        target_modules = ["q_proj", "v_proj", "gate_proj"],
        modules_to_save = ["embed_tokens", "lm_head"],
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": True})
    # Reentrant checkpointing needs a grad-requiring input (Unsloth's GPU path does the same).
    model.enable_input_require_grads()
    model.train()
    args = _args()
    find_unused = _helper()(args, model)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    if distributed:
        # As transformers' Trainer builds it: find_unused_parameters from the args, True when None.
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters = True if find_unused is None else find_unused
        )
    optimizer = torch.optim.SGD(params, lr = 0.5)

    g = torch.Generator().manual_seed(1)
    losses, grad_norms = [], []
    for step in range(STEPS):
        ids = torch.randint(1, 60, (GLOBAL_BATCH, 12), generator = g)
        local = ids.chunk(world)[rank]
        loss = model(input_ids = local, labels = local).loss
        loss.backward()
        total = loss.detach().clone()
        if distributed:
            dist.all_reduce(total)
            total /= world
        losses.append(total.item())
        if step == 0:
            grad_norms = [p.grad.norm().item() for p in params]
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)

    os.makedirs(out_dir, exist_ok = True)
    with open(os.path.join(out_dir, f"result_{rank}.json"), "w") as f:
        json.dump({"names": names, "losses": losses, "grad_norms": grad_norms}, f)
    print(f"RANK_OK find_unused={find_unused}", flush = True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1])
