# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""One gloo rank for test_grpo_ddp_gradient_sync.py. No unsloth import at module level: a spawned rank must set up the GPU-free harness first."""

from __future__ import annotations

import os
import runpy

MODES = ("bypass", "through_ddp", "no_sync")


def run_rank(rank, port, queue):
    runpy.run_path(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "conftest.py")
    )
    import torch
    import torch.distributed as dist
    import unsloth.models.rl as rl

    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.body = torch.nn.Linear(6, 6)
            self.head = torch.nn.Linear(6, 11, bias = False)

        def forward(self, x):
            return self.head(self.body(x))

    class Trainer:
        """The generated GRPO trainer's loss runs the UNWRAPPED module (zoo's grpo_accumulated_loss)."""

        def __init__(self, through_ddp):
            self.through_ddp = through_ddp

        def training_step(self, model, inputs):
            forward = model if self.through_ddp else model.module
            loss = forward(inputs).logsumexp(-1).mean()
            loss.backward()
            return loss.detach()

    rl._wrap_grpo_ddp_gradient_sync(Trainer)
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    dist.init_process_group("gloo", rank = rank, world_size = 2)
    try:
        results = {}
        x = torch.randn(3, 4, 6, generator = torch.Generator().manual_seed(100 + rank))
        local = LM()
        local(x).logsumexp(-1).mean().backward()
        local_grads = torch.cat([p.grad.reshape(-1) for p in local.parameters()])
        gathered = [torch.zeros_like(local_grads) for _ in range(2)]
        dist.all_gather(gathered, local_grads)
        for mode in MODES:
            ddp = torch.nn.parallel.DistributedDataParallel(LM())
            trainer = Trainer(through_ddp = mode == "through_ddp")
            if mode == "no_sync":
                with ddp.no_sync():
                    trainer.training_step(ddp, x)
            else:
                trainer.training_step(ddp, x)
            results[mode] = torch.cat(
                [p.grad.reshape(-1) for p in ddp.module.parameters()]
            ).tolist()
        # bucket_bytes=1: every gradient is larger than a bucket, so each one is reduced in place.
        ddp = torch.nn.parallel.DistributedDataParallel(LM())
        ddp.module(x).logsumexp(-1).mean().backward()
        rl._unsloth_average_gradients(ddp, bucket_bytes = 1)
        results["oversized"] = torch.cat(
            [p.grad.reshape(-1) for p in ddp.module.parameters()]
        ).tolist()
        queue.put((rank, results, torch.stack(gathered).mean(0).tolist(), local_grads.tolist()))
    finally:
        dist.destroy_process_group()
