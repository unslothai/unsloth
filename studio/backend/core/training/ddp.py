# SPDX-License-Identifier: AGPL-3.0-only
"""Single-node DDP launcher for Studio text training."""

from __future__ import annotations

import copy
import multiprocessing as mp
import os
import queue
import socket
import subprocess
import threading
import time
import traceback
from typing import Any


def _nvidia_smi_child_env() -> dict[str, str]:
    from utils.native_path_leases import child_env_without_native_path_secret
    return child_env_without_native_path_secret()


def training_precision_flags_for_dtype(
    ddp_dtype: str | None, supports_bf16: bool
) -> dict[str, bool]:
    """Resolve mutually exclusive Trainer flags from the DDP-wide dtype.

    The local capability can differ by rank (e.g. RTX 3070 supports native BF16,
    while a paired RTX 2080 Ti does not), so the coordinator override takes
    precedence over Unsloth's per-process capability result.
    """
    if ddp_dtype == "fp16":
        return {"fp16": True, "bf16": False}
    if ddp_dtype == "bf16":
        return {"fp16": False, "bf16": True}
    return {"fp16": not supports_bf16, "bf16": supports_bf16}


def model_load_dtype_for_training(
    ddp_dtype: str | None, is_rocm: bool, supports_bf16: bool
) -> str | None:
    """Choose the model loader dtype to match DDP Trainer precision."""
    if ddp_dtype in ("fp16", "bf16"):
        return ddp_dtype
    if is_rocm and not supports_bf16:
        return "fp16"
    return None


def _common_cuda_dtype(gpu_ids: list[int]) -> str:
    """Choose one dtype the whole DDP process group can run natively.

    DDP replicates the model on every rank, so using each GPU's local preference
    can produce incompatible Trainer configs. Turing (SM 7.x), for example,
    requires FP16 even when another selected rank is Ampere or newer.
    """
    try:
        from utils.hardware import gpu_query

        result = gpu_query.run_nvidia_smi(
            ["nvidia-smi", "--query-gpu=index,compute_cap", "--format=csv,noheader"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 10,
            cache = False,
            env = _nvidia_smi_child_env(),
        )
        if result.returncode != 0:
            return "fp16"
        capabilities = {}
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2:
                continue
            try:
                major, minor = parts[1].split(".", 1)
                capabilities[int(parts[0])] = (int(major), int(minor))
            except ValueError:
                continue
        # A missing probe result must not let one rank independently choose BF16.
        if not gpu_ids or any(gpu_id not in capabilities for gpu_id in gpu_ids):
            return "fp16"
        return "bf16" if all(capabilities[gpu_id] >= (8, 0) for gpu_id in gpu_ids) else "fp16"
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        print(f"DDP GPU capability probe failed; using common fp16: {exc}", flush = True)
        return "fp16"


class _RankEvents:
    """Forward rank-zero progress and make errors from any rank fatal to spawn."""

    def __init__(self, rank: int, event_queue: Any, failures: Any):
        self.rank = rank
        self.event_queue = event_queue
        self.failures = failures
        self.error: dict[str, Any] | None = None

    def put(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        if event.get("type") == "error":
            if self.error is not None:
                return
            self.error = dict(event)
            self.failures[self.rank] = self.error
        elif self.rank == 0:
            self.event_queue.put(event)


class _SharedStopQueue:
    def __init__(self, stop_event: Any, save_value: Any):
        self._stop_event = stop_event
        self._save_value = save_value
        self._delivered = False

    def get(self, timeout: float | None = None) -> dict[str, Any]:
        if self._delivered or not self._stop_event.wait(timeout):
            raise queue.Empty
        self._delivered = True
        return {"type": "stop", "save": bool(self._save_value.value)}


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _rank_entry(
    rank: int,
    config: dict[str, Any],
    event_queue: Any,
    stop_event: Any,
    save_value: Any,
    master_port: int,
    failures: Any,
) -> None:
    gpu_ids = list(config.get("resolved_gpu_ids") or [])
    physical_gpu = int(gpu_ids[rank])
    world_size = len(gpu_ids)
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(master_port),
        }
    )
    # Set before importing the worker/Unsloth: their BF16 capability is initialized
    # at import time, and the coordinator's common dtype must win on every rank.
    os.environ["UNSLOTH_DDP_COMMON_DTYPE"] = str(config.get("_ddp_common_dtype", "fp16"))
    rank_config = copy.deepcopy(config)
    rank_config["_ddp_child"] = True
    rank_config["_ddp_rank"] = rank
    rank_config["_ddp_world_size"] = world_size
    rank_config["resolved_gpu_ids"] = [physical_gpu]
    from .worker import run_training_process

    rank_events = _RankEvents(rank, event_queue, failures)
    run_training_process(
        event_queue = rank_events,
        stop_queue = _SharedStopQueue(stop_event, save_value),
        config = rank_config,
    )
    if rank_events.error is not None:
        raise RuntimeError(
            f"DDP rank {rank} failed: " f"{rank_events.error.get('error') or 'Training failed'}"
        )


def run_ddp_training_process(*, event_queue: Any, stop_queue: Any, config: dict[str, Any]) -> None:
    """Launch one worker rank per selected NVIDIA GPU and relay rank-zero events."""
    gpu_ids = list(config.get("resolved_gpu_ids") or [])
    if len(gpu_ids) < 2:
        raise ValueError("DDP requires at least two resolved GPU ids.")
    config = copy.deepcopy(config)
    config["_ddp_common_dtype"] = _common_cuda_dtype(gpu_ids)

    manager = mp.Manager()
    rank_events = manager.Queue()
    stop_event = manager.Event()
    save_value = manager.Value("b", True)
    failures = manager.dict()
    done = threading.Event()

    def relay_events() -> None:
        while not done.is_set() or not rank_events.empty():
            try:
                event = rank_events.get(timeout = 0.2)
            except queue.Empty:
                if done.is_set():
                    return
                continue
            if event is None:
                return
            event_queue.put(event)

    def relay_stop() -> None:
        while not done.is_set():
            try:
                message = stop_queue.get(timeout = 0.2)
            except queue.Empty:
                continue
            if isinstance(message, dict) and message.get("type") == "stop":
                save_value.value = bool(message.get("save", True))
                stop_event.set()
                return

    event_thread = threading.Thread(target = relay_events, daemon = True)
    stop_thread = threading.Thread(target = relay_stop, daemon = True)
    event_thread.start()
    stop_thread.start()
    try:
        import torch.multiprocessing as torch_mp
        torch_mp.spawn(
            _rank_entry,
            args = (
                config,
                rank_events,
                stop_event,
                save_value,
                _free_loopback_port(),
                failures,
            ),
            nprocs = len(gpu_ids),
            join = True,
        )
    except BaseException as exc:
        if failures:
            failed_rank = min(failures)
            error = failures[failed_rank]
            event_queue.put(
                {
                    "type": "error",
                    "error": f"DDP rank {failed_rank} failed: "
                    f"{error.get('error') or 'Training failed'}",
                    "stack": error.get("stack", ""),
                    "ts": error.get("ts", time.time()),
                }
            )
        else:
            event_queue.put(
                {
                    "type": "error",
                    "error": f"DDP worker group failed: {exc}",
                    "stack": traceback.format_exc(limit = 20),
                    "ts": time.time(),
                }
            )
    finally:
        done.set()
        event_thread.join(timeout = 2.0)
        stop_thread.join(timeout = 1.0)
        manager.shutdown()
