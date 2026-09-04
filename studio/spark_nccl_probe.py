# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measure real NCCL all-reduce bandwidth between two DGX Sparks.

Run under torchrun on both nodes; rank 0 prints one machine-readable line:

    SPARK_NCCL_BUSBW <GB/s> <ms>

Why this exists rather than trusting `ib_write_bw`: on GB10 the raw RDMA number stays
healthy (~24.5 GB/s) even when NCCL has collapsed to ~3 GB/s. We hit exactly that, and
chased it for a long time, because every layer we could measure cheaply looked fine. Only
a real collective shows the fault, so that is what this runs.

Deliberately not a general benchmark: one message size, few iterations, no warmup beyond
what is needed to build the communicator. It has to be quick enough to sit behind a
`unsloth spark doctor` that someone runs while wondering why training is slow.
"""

from __future__ import annotations

import os
import time


def main() -> int:
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("nccl")

    mb = int(os.environ.get("SPARK_PROBE_MB", "1024"))
    iters = int(os.environ.get("SPARK_PROBE_ITERS", "6"))
    buf = torch.empty(mb * 1024 * 1024 // 4, dtype = torch.float32, device = "cuda")

    for _ in range(2):                      # build the communicator, touch every channel
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    dist.barrier()

    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    dist.barrier()
    per_iter = (time.perf_counter() - t0) / iters

    if rank == 0:
        gib = buf.numel() * 4 / 2 ** 30
        # Standard nccl-tests bus-bandwidth convention, so the number is directly
        # comparable to published all_reduce figures.
        busbw = gib * 2 * (world - 1) / world / per_iter
        print(f"SPARK_NCCL_BUSBW {busbw:.2f} {per_iter * 1000:.1f}", flush = True)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
