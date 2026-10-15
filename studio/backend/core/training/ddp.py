# SPDX-License-Identifier: AGPL-3.0-only
"""Single-node DDP launcher for Studio text training."""

from __future__ import annotations

import copy
import os
import queue
import socket
import subprocess
import threading
import time
import traceback
from typing import Any


def _wait_for_stop(stop_event: Any, timeout: float | None) -> bool:
    if timeout is None:
        stop_event.wait()
        return True
    if timeout < 0:
        raise ValueError("'timeout' must be a non-negative number")
    if timeout == 0:
        return bool(stop_event.is_set())
    if stop_event.wait(timeout):
        return True
    return False


def _nvidia_smi_child_env() -> dict[str, str]:
    from utils.native_path_leases import child_env_without_native_path_secret
    return child_env_without_native_path_secret()


def training_precision_flags_for_dtype(
    ddp_dtype: str | None, supports_bf16: bool
) -> dict[str, bool]:
    """The coordinator dtype overrides each rank's local capability, which can differ."""
    if ddp_dtype == "fp16":
        return {"fp16": True, "bf16": False}
    if ddp_dtype == "bf16":
        return {"fp16": False, "bf16": True}
    return {"fp16": not supports_bf16, "bf16": supports_bf16}


def model_load_dtype_for_training(
    ddp_dtype: str | None, is_rocm: bool, supports_bf16: bool
) -> str | None:
    if ddp_dtype in ("fp16", "bf16"):
        return ddp_dtype
    if is_rocm and not supports_bf16:
        return "fp16"
    return None


def _common_cuda_dtype(gpu_ids: list[int]) -> str:
    """One dtype for every rank: Turing (SM 7.x) needs FP16 even beside Ampere ranks."""
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
        # CUDA ids match nvidia-smi only under CUDA_DEVICE_ORDER=PCI_BUS_ID; otherwise judge every card.
        if os.environ.get("CUDA_DEVICE_ORDER", "PCI_BUS_ID") != "PCI_BUS_ID":
            gpu_ids = list(capabilities)
        return "bf16" if all(capabilities[gpu_id] >= (8, 0) for gpu_id in gpu_ids) else "fp16"
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError) as exc:
        print(f"DDP GPU capability probe failed; using common fp16: {exc}", flush = True)
        return "fp16"


# The backend arms Xet->HTTP recovery only inside a model load, so the load window must span every rank's.
_MODEL_LOAD_EVENTS = ("model_load_started", "model_load_resolved", "model_load_completed")


class _RankEvents:
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
        elif event.get("type") in _MODEL_LOAD_EVENTS:
            self.event_queue.put({**event, "_ddp_rank": self.rank})
        elif self.rank == 0 or event.get("type") == "stall":
            self.event_queue.put(event)


class _SharedStopQueue:
    def __init__(self, stop_event: Any, cancel_event: Any):
        self._stop_event = stop_event
        self._cancel_event = cancel_event
        self._stop_delivered = False
        self._cancel_delivered = False

    def get(self, timeout: float | None = None) -> dict[str, Any]:
        if not self._stop_delivered:
            if not _wait_for_stop(self._stop_event, timeout):
                raise queue.Empty
            self._stop_delivered = True
            self._cancel_delivered = bool(self._cancel_event.is_set())
            return {"type": "stop", "save": not self._cancel_delivered}
        if not self._cancel_delivered:
            if not _wait_for_stop(self._cancel_event, timeout):
                raise queue.Empty
            self._cancel_delivered = True
            return {"type": "stop", "save": False}
        # Queue.get semantics: once both are consumed, later polls still wait out their timeout.
        _wait_for_stop(threading.Event(), timeout)
        raise queue.Empty


def _bind_to_parent_lifetime() -> None:
    """SIGTERM on parent death: SIGINT cannot interrupt a rank blocked in a collective."""
    try:
        from utils.process_lifetime import bind_current_process_to_parent_lifetime
        bind_current_process_to_parent_lifetime()
    except Exception:
        pass


def _destroy_process_group(synchronize: bool) -> None:
    """Leave together: rank 0 hosts the store, so exiting first strands its peers."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            if synchronize:
                dist.barrier()
            dist.destroy_process_group()
    except Exception:
        pass


def _start_manager() -> Any:
    from multiprocessing.managers import SyncManager

    # A force-terminated coordinator runs no cleanup, so the manager binds itself to it.
    manager = SyncManager()
    manager.start(initializer = _bind_to_parent_lifetime)
    return manager


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _rank_entry(
    rank: int,
    config: dict[str, Any],
    event_queue: Any,
    stop_event: Any,
    cancel_event: Any,
    master_port: int,
    failures: Any,
) -> None:
    _bind_to_parent_lifetime()
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
    # Before importing worker/Unsloth: BF16 capability is fixed at import time.
    os.environ["UNSLOTH_DDP_COMMON_DTYPE"] = str(config.get("_ddp_common_dtype", "fp16"))
    rank_config = copy.deepcopy(config)
    rank_config["_ddp_child"] = True
    rank_config["_ddp_rank"] = rank
    rank_config["_ddp_world_size"] = world_size
    rank_config["resolved_gpu_ids"] = [physical_gpu]
    account = rank_config.pop("_ddp_account", None)
    from core.training.account_jobs import run_account_child
    from utils.account_context import OWNER

    rank_events = _RankEvents(rank, event_queue, failures)
    finished = False
    try:
        # Like account_process_spec: ranks act as the managed account, not the owner.
        run_account_child(
            account = account or OWNER,
            job_module = "core.training.worker",
            job_target = "run_training_process",
            event_queue = rank_events,
            stop_queue = _SharedStopQueue(stop_event, cancel_event),
            config = rank_config,
        )
        finished = rank_events.error is None
    finally:
        # A failed rank must not wait in a barrier its peers never reach.
        _destroy_process_group(synchronize = finished)
    if rank_events.error is not None:
        raise RuntimeError(
            f"DDP rank {rank} failed: " f"{rank_events.error.get('error') or 'Training failed'}"
        )


# A stop before the process group exists lets a rank return while a peer waits at the rendezvous.
_STOPPED_PEER_GRACE_S = 30.0


def _join_ranks(context: Any, stop_event: Any) -> None:
    """Join every rank; once a stop is requested and a rank has exited cleanly, end peers that
    are still waiting for it after a grace period instead of until the rendezvous times out."""
    first_exit = None
    while not context.join(timeout = 1.0):
        if not stop_event.is_set():
            continue
        alive = [process for process in context.processes if process.is_alive()]
        if not alive or len(alive) == len(context.processes):
            continue
        first_exit = first_exit or time.monotonic()
        if time.monotonic() - first_exit < _STOPPED_PEER_GRACE_S:
            continue
        for process in alive:
            process.terminate()
        for process in alive:
            process.join(timeout = 10.0)
            if process.is_alive():
                process.kill()
                process.join(timeout = 5.0)
        return


def run_ddp_training_process(*, event_queue: Any, stop_queue: Any, config: dict[str, Any]) -> None:
    gpu_ids = list(config.get("resolved_gpu_ids") or [])
    if len(gpu_ids) < 2:
        raise ValueError("DDP requires at least two resolved GPU ids.")
    config = copy.deepcopy(config)
    config["_ddp_common_dtype"] = _common_cuda_dtype(gpu_ids)
    from utils.account_context import current_account

    config["_ddp_account"] = current_account()
    if not config.get("output_dir") and not config.get("resume_from_checkpoint"):
        from utils.training_runs import build_default_output_dir_name

        # Ranks reach the worker's default at different times and would stamp different dirs.
        config["output_dir"] = build_default_output_dir_name(
            config["model_name"], config.get("project_name")
        )

    manager = _start_manager()
    rank_events = manager.Queue()
    stop_event = manager.Event()
    cancel_event = manager.Event()
    failures = manager.dict()
    done = threading.Event()
    # Rank zero finishes before its peers: completion waits for the whole group to succeed.
    held_complete: list[dict[str, Any]] = []

    loading_ranks: set[int] = set()

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
            if isinstance(event, dict) and event.get("type") == "complete":
                held_complete[:] = [event]
                continue
            if isinstance(event, dict) and event.get("type") in _MODEL_LOAD_EVENTS:
                event = dict(event)
                rank = event.pop("_ddp_rank", 0)
                if event["type"] == "model_load_started":
                    was_loading = bool(loading_ranks)
                    loading_ranks.add(rank)
                    if was_loading:
                        continue
                elif event["type"] == "model_load_completed":
                    loading_ranks.discard(rank)
                    if loading_ranks:
                        continue
            event_queue.put(event)

    def relay_stop() -> None:
        while not done.is_set():
            try:
                message = stop_queue.get(timeout = 0.2)
            except queue.Empty:
                continue
            except (EOFError, OSError, ValueError):
                return
            if isinstance(message, dict) and message.get("type") == "stop":
                if not message.get("save", True):
                    cancel_event.set()
                stop_event.set()
                if cancel_event.is_set():
                    return

    event_thread = threading.Thread(target = relay_events, daemon = True)
    stop_thread = threading.Thread(target = relay_stop, daemon = True)
    event_thread.start()
    stop_thread.start()
    try:
        import torch.multiprocessing as torch_mp
        context = torch_mp.spawn(
            _rank_entry,
            args = (
                config,
                rank_events,
                stop_event,
                cancel_event,
                _free_loopback_port(),
                failures,
            ),
            nprocs = len(gpu_ids),
            join = False,
        )
        _join_ranks(context, stop_event)
    except BaseException as exc:
        held_complete.clear()
        if failures:
            failed_rank = min(failures)
            error = failures[failed_rank]
            terminal = {
                "type": "error",
                "error": f"DDP rank {failed_rank} failed: {error.get('error') or 'Training failed'}",
                "stack": error.get("stack", ""),
                "ts": error.get("ts", time.time()),
            }
        else:
            terminal = {
                "type": "error",
                "error": f"DDP worker group failed: {exc}",
                "stack": traceback.format_exc(limit = 20),
                "ts": time.time(),
            }
    else:
        terminal = None
    finally:
        done.set()
        # Every rank has exited, so the queue is final: drain it before the terminal event.
        event_thread.join(timeout = 30.0)
        stop_thread.join(timeout = 1.0)
        manager.shutdown()
    if terminal is None and held_complete:
        terminal = held_complete[-1]
    if terminal is not None:
        event_queue.put(terminal)
