# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multi-process training ranks for the Studio worker, launched the way torchrun would.

The training worker Studio already spawns is rank 0. Before torch or unsloth is imported it sets the torchrun
environment (RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR / MASTER_PORT on loopback) and UNSLOTH_PARALLEL, then starts
ranks 1..N-1 as its own child processes running the same worker entry on the same config. Unsloth builds the mesh
from that environment. Only rank 0 talks to the UI: the other ranks' events go to the log with a rank prefix.
Stop: rank 0 broadcasts (stop, save) at every step end (`sync_stop`), so every rank leaves the loop at the same step.
A rank that dies ends the run (rank 0 reports it); a rank whose parent is gone exits within 2 s; cancel kills the
worker's process tree, which holds every rank.
"""

from __future__ import annotations

import atexit
import os
import pickle
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Optional

__all__ = [
    "CHILD_ENV",
    "launch_ranks",
    "rank_main",
    "rank",
    "world_size",
    "is_main_rank",
    "is_multi_process",
    "sync_stop",
]

CHILD_ENV = "UNSLOTH_STUDIO_PARALLEL_RANK"  # set in ranks 1..N-1
_WORLD_ENV = "UNSLOTH_STUDIO_PARALLEL_WORLD"  # set in every rank of a Studio multi-process run
_PARENT_ENV = "UNSLOTH_STUDIO_PARALLEL_PARENT"
# module:function ranks 1..N-1 run (tests point it at a small target).
_ENTRY_ENV = "UNSLOTH_STUDIO_PARALLEL_ENTRY"
_DEFAULT_ENTRY = "core.training.worker:run_training_process"
_HOST = "127.0.0.1"
_PROCS: list = []
_CLOSING = threading.Event()


def rank() -> int:
    try:
        return int(os.environ.get(CHILD_ENV, "0") or 0)
    except ValueError:
        return 0


def world_size() -> int:
    try:
        return max(1, int(os.environ.get(_WORLD_ENV, "1") or 1))
    except ValueError:
        return 1


def is_multi_process() -> bool:
    return world_size() > 1


def is_main_rank() -> bool:
    return rank() == 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((_HOST, 0))
        return s.getsockname()[1]


def _backend_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _rank_env(base: dict, r: int, world: int) -> dict:
    env = dict(base)
    env.update(
        RANK = str(r),
        LOCAL_RANK = str(r),
        GROUP_RANK = "0",
        ROLE_RANK = str(r),
        WORLD_SIZE = str(world),
        LOCAL_WORLD_SIZE = str(world),
    )
    return env


def _pump(stream, prefix: str) -> None:
    """Child output -> this process's stderr (the server log), one prefixed line at a time."""
    try:
        for raw in iter(stream.readline, b""):
            try:
                sys.stderr.write(prefix + raw.decode("utf-8", "replace"))
                sys.stderr.flush()
            except Exception:  # noqa: BLE001 -- a closed log must not stop the rank
                pass
    finally:
        try:
            stream.close()
        except Exception:  # noqa: BLE001
            pass


def _terminate_children() -> None:
    _CLOSING.set()
    for proc in _PROCS:
        if proc.poll() is None:
            try:
                proc.terminate()
            except Exception:  # noqa: BLE001
                pass
    deadline = time.monotonic() + 5
    for proc in _PROCS:
        try:
            proc.wait(timeout = max(0.1, deadline - time.monotonic()))
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


def launch_ranks(
    config: dict,
    event_queue: Any,
    *,
    python: Optional[str] = None,
) -> int:
    """In rank 0, before torch / unsloth import: export the torchrun env, start ranks 1..N-1. Returns N."""
    plan = config.get("parallel") or {}
    world = int(plan.get("world_size") or 1)
    if world <= 1 or plan.get("launcher") != "multi_process":
        return 1
    if os.environ.get(CHILD_ENV):
        return world  # ranks 1..N-1 run the same entry; only rank 0 starts ranks
    port = _free_port()
    os.environ.update(
        MASTER_ADDR = _HOST,
        MASTER_PORT = str(port),
        **{_WORLD_ENV: str(world)},
    )
    os.environ.update(_rank_env(os.environ, 0, world))
    os.environ.pop(CHILD_ENV, None)
    if plan.get("spec"):
        os.environ["UNSLOTH_PARALLEL"] = str(plan["spec"])
    # Every rank is on this host: NCCL's bootstrap needs loopback only (a user's own NCCL_SOCKET_IFNAME wins).
    if sys.platform.startswith("linux"):
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join([_backend_dir(), *filter(None, [base.get("PYTHONPATH")])])
    base[_PARENT_ENV] = str(os.getpid())
    base["PYTHONUNBUFFERED"] = "1"
    child_config = dict(config)
    # The mask rank 0 narrowed is inherited; re-applying physical ids to it would narrow twice.
    child_config["resolved_gpu_ids"] = None
    payload = pickle.dumps(child_config)
    for r in range(1, world):
        env = _rank_env(base, r, world)
        env[CHILD_ENV] = str(r)
        proc = subprocess.Popen(
            [
                python or sys.executable,
                "-c",
                "from core.training.parallel_launch import rank_main; rank_main()",
            ],
            stdin = subprocess.PIPE,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            env = env,
            cwd = os.getcwd(),
        )
        proc.stdin.write(payload)
        proc.stdin.close()
        _PROCS.append(proc)
        threading.Thread(target = _pump, args = (proc.stdout, f"[rank {r}] "), daemon = True).start()
    atexit.register(_terminate_children)

    def _watch():
        while not _CLOSING.is_set():
            for r, proc in enumerate(_PROCS, start = 1):
                code = proc.poll()
                if code not in (None, 0) and not _CLOSING.is_set():
                    message = (
                        f"Multi-GPU training: rank {r} of {world} exited with code {code}; stopping the run. "
                        "The server log has that rank's error (lines starting with [rank "
                        f"{r}])."
                    )
                    try:
                        event_queue.put(
                            {"type": "error", "error": message, "stack": "", "ts": time.time()}
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    _CLOSING.set()
                    time.sleep(1.5)  # let the queue's feeder thread flush the event
                    _terminate_children()
                    os._exit(1)
            time.sleep(0.5)

    threading.Thread(target = _watch, name = "studio-parallel-watch", daemon = True).start()
    return world


class _RankEvents:
    """Event sink of ranks 1..N-1: errors and warnings reach the log, nothing reaches the UI."""

    def __init__(self, r: int):
        self.rank = r

    def put(self, event: Any, *args, **kwargs) -> None:
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind in ("error", "warning"):
            text = event.get("error") or event.get("message") or ""
            print(f"{kind}: {text}", file = sys.stderr, flush = True)
            if kind == "error" and event.get("stack"):
                print(event["stack"], file = sys.stderr, flush = True)

    def put_nowait(self, event: Any) -> None:
        self.put(event)


class _NoStop:
    """Stop queue of ranks 1..N-1: never yields; rank 0's broadcast stops them."""

    def get(
        self,
        timeout: Optional[float] = None,
        *args,
        **kwargs,
    ):
        import queue
        time.sleep(timeout if timeout is not None else 3600)
        raise queue.Empty

    def get_nowait(self):
        import queue
        raise queue.Empty

    def put(self, *args, **kwargs) -> None:
        return None


def rank_main() -> None:
    """Entry of ranks 1..N-1: config from stdin, the same worker entry as rank 0, exit without NCCL teardown."""
    r = rank()
    parent = int(os.environ.get(_PARENT_ENV, "0") or 0)

    def _parent_watch():
        while True:
            if parent and os.getppid() != parent:
                os._exit(1)
            time.sleep(1)

    threading.Thread(target = _parent_watch, name = "studio-parallel-parent", daemon = True).start()
    backend_dir = _backend_dir()
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)
    try:
        from utils.process_lifetime import bind_current_process_to_parent_lifetime
        bind_current_process_to_parent_lifetime()
    except Exception:  # noqa: BLE001 -- the parent watch above still ends an orphaned rank
        pass
    config = pickle.loads(sys.stdin.buffer.read())
    code = 0
    try:
        import importlib

        module, _, name = (os.environ.get(_ENTRY_ENV) or _DEFAULT_ENTRY).partition(":")
        target = getattr(importlib.import_module(module), name)
        target(event_queue = _RankEvents(r), stop_queue = _NoStop(), config = config)
    except BaseException:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    # No communicator teardown: NCCL's finalize waits on peers that may already be gone (rank 0 may still be saving).
    os._exit(code)


def sync_stop(should_stop: bool, save_on_stop: bool) -> tuple:
    """Every rank's view of rank 0's (stop, save) at this step. One 2-int broadcast; a no-op outside a Studio
    multi-process run or before the process group exists."""
    if not is_multi_process():
        return should_stop, save_on_stop
    try:
        import torch
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            return should_stop, save_on_stop
        device = "cpu"
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        flags = torch.tensor(
            [int(bool(should_stop)), int(bool(save_on_stop))], dtype = torch.int32, device = device
        )
        dist.broadcast(flags, src = 0)
        values = flags.tolist()
        return bool(values[0]), bool(values[1])
    except Exception:  # noqa: BLE001 -- a failed broadcast must not hide rank 0's own stop
        return should_stop, save_on_stop
