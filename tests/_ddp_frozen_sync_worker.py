# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""One gloo rank for test_ddp_frozen_sync.py: a real accelerate Accelerator on CPU builds the DDP."""

from __future__ import annotations

import os
import types

from _ddp_comm_defaults_worker import load_defaults


def build(rank, peft):
    import torch

    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Embedding(32, 8), torch.nn.Linear(8, 4))
    model[0].weight.requires_grad_(False)
    with torch.no_grad():
        # Marks which rank a weight came from: DDP's start broadcast would overwrite rank 1's with rank 0's.
        model[0].weight.fill_(float(rank))
        model[1].weight.fill_(float(rank))
    if peft:
        model.peft_config = {"default": object()}
    return model


def run_rank(rank, port, queue, cases):
    import torch

    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        LOCAL_RANK = str(rank),
        WORLD_SIZE = "2",
        LOCAL_WORLD_SIZE = "2",
        ACCELERATE_USE_CPU = "true",
    )
    torch.set_num_threads(1)
    from accelerate import Accelerator

    out = {}
    accelerator = Accelerator(cpu = True)
    try:
        defaults = load_defaults()
        for name, peft, env in cases:
            saved = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                accelerator.ddp_handler = None
                accelerator.prepare_model = types.MethodType(
                    type(accelerator).prepare_model, accelerator
                )
                trainer = types.SimpleNamespace(
                    accelerator = accelerator, args = types.SimpleNamespace()
                )
                defaults.install_unsloth_ddp_comm_defaults(trainer)
                import torch._dynamo

                torch._dynamo.config.optimize_ddp = True  # torch's default
                ddp = accelerator.prepare_model(build(rank, peft))
                x = torch.tensor([[1, 2, 3]])
                ddp(x).sum().backward()
                out[name] = dict(
                    frozen = float(ddp.module[0].weight.mean()),
                    trainable = float(ddp.module[1].weight.mean()),
                    grad = ddp.module[1].weight.grad.tolist(),
                    optimize_ddp = torch._dynamo.config.optimize_ddp,
                )
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        queue.put((rank, out))
    except BaseException:
        import traceback
        queue.put((rank, {"error": traceback.format_exc()}))
