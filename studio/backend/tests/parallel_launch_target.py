# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A small training-worker stand-in for tests/test_parallel_launch.py: same entry signature as
core.training.worker.run_training_process, a gloo process group, one sync_stop per step."""

import json
import os
import queue
import time


def train(event_queue, stop_queue, config):
    from core.training import parallel_launch

    # Rank 0 is started like the Studio worker; it starts the others exactly as the worker does.
    parallel_launch.launch_ranks(config, event_queue)
    import torch.distributed as dist

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    info = {
        key: os.environ.get(key)
        for key in (
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "UNSLOTH_PARALLEL",
        )
    }
    info["resolved_gpu_ids"] = config.get("resolved_gpu_ids")
    info["is_main_rank"] = parallel_launch.is_main_rank()
    stop = save = False
    step = 0
    for step in range(1, int(config.get("steps", 1000)) + 1):
        if rank == 0:
            try:
                msg = stop_queue.get_nowait()
                stop, save = True, bool(msg.get("save", True))
            except queue.Empty:
                pass
        if config.get("fail_rank") == rank and step == config.get("fail_step"):
            os._exit(3)
        stop, save = parallel_launch.sync_stop(stop, save)
        event_queue.put({"type": "progress", "step": step, "rank": rank})
        if stop:
            break
        time.sleep(0.2)
    info.update(last_step = step, stopped = stop, save = save)
    with open(os.path.join(config["out_dir"], f"rank{rank}.json"), "w") as f:
        json.dump(info, f)
    dist.barrier()
    event_queue.put({"type": "complete", "rank": rank})
