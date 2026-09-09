# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measure real NCCL all-reduce bandwidth between two DGX Sparks.

Run under torchrun on both nodes; rank 0 prints `SPARK_NCCL_BUSBW <GB/s> <ms>`.

Not `ib_write_bw`: on GB10 the raw RDMA number stays healthy even when NCCL has collapsed,
so only a real collective shows the fault. Not a general benchmark either -- one message
size and few iterations, so it can sit behind a `doctor` someone runs impatiently.
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

    for _ in range(2):  # build the communicator, touch every channel
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
        gib = buf.numel() * 4 / 2**30
        # The nccl-tests bus-bandwidth convention, so this is comparable to published figures.
        busbw = gib * 2 * (world - 1) / world / per_iter
        print(f"SPARK_NCCL_BUSBW {busbw:.2f} {per_iter * 1000:.1f}", flush = True)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
