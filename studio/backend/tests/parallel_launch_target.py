# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training-worker stand-in for test_parallel_launch.py: gloo group, one sync_stop per step."""

import json
import os
import queue
import time


def train(event_queue, stop_queue, config):
    from core.training import parallel_launch

    parallel_launch.launch_ranks(config, event_queue, entry = ("parallel_launch_target", "train"))
    import torch.distributed as dist

    assert dist.is_initialized() and dist.get_world_size() == 2
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
            "UNSLOTH_FSDP2_LAYOUT",
            "UNSLOTH_STUDIO_TEST_NOT_ALLOWED",
        )
    }
    from core.training import worker

    info["data_parallel_world_size"] = worker._data_parallel_world_size()
    info["resolved_gpu_ids"] = config.get("resolved_gpu_ids")
    info["output_dir"] = config.get("output_dir")
    from utils.account_context import current_account

    info["account_id"] = current_account().account_id
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
            if config.get("fail_by_event"):
                event_queue.put({"type": "error", "error": "model load failed"})
                return
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


def train_as(event_queue, stop_queue, config):
    from utils.account_context import AccountContext, run_as
    account = AccountContext(**config["as_account"])
    return run_as(account, train, event_queue = event_queue, stop_queue = stop_queue, config = config)
