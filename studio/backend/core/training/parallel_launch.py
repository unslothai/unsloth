# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Torchrun-style training ranks: the Studio worker is rank 0 and spawns ranks 1..N-1 on the same config.

The process group's store listens on loopback only (a plain env:// store would listen on every interface).
Only rank 0 talks to the UI. Rank 0 broadcasts (stop, save) via `sync_stop` so every rank stops at the same step.
"""

from __future__ import annotations

import atexit
import dataclasses
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import timedelta
from typing import Any, Callable, Optional

__all__ = [
    "CHILD_ENV",
    "launch_ranks",
    "rank_main",
    "rank",
    "world_size",
    "data_parallel_size",
    "is_main_rank",
    "is_multi_process",
    "sync_stop",
    "wait_for_ranks",
]

CHILD_ENV = "UNSLOTH_STUDIO_PARALLEL_RANK"
_WORLD_ENV = "UNSLOTH_STUDIO_PARALLEL_WORLD"
_PARENT_ENV = "UNSLOTH_STUDIO_PARALLEL_PARENT"
CACHE_CLEARED_ENV = "UNSLOTH_STUDIO_COMPILED_CACHE_CLEARED"
TIMEOUT_ENV = "UNSLOTH_STUDIO_PARALLEL_TIMEOUT_S"
_DEFAULT_ENTRY = ("core.training.worker", "run_training_process")
# Stale layout vars in the server env would conflict with UNSLOTH_PARALLEL.
_LAYOUT_ENV = (
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_CP_SIZE",
    "UNSLOTH_CP_BACKEND",
    "UNSLOTH_EP_SIZE",
    "UNSLOTH_FSDP2",
    "UNSLOTH_FSDP2_BACKEND",
    "UNSLOTH_PP_SCHEDULE",
    "UNSLOTH_PP_MICROBATCHES",
    "UNSLOTH_PARALLEL_AUTO_RESOLVED",
)
_PLAN_ENV = ("UNSLOTH_FSDP2_LAYOUT",)
_HOST = "127.0.0.1"
_MODULE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
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


def data_parallel_size() -> int:
    world = world_size()
    if world <= 1:
        return 1
    model_ranks = 1
    for part in str(os.environ.get("UNSLOTH_PARALLEL", "")).replace(";", ",").split(","):
        key, _, value = (x.strip().lower() for x in part.partition("="))
        if key in ("tp", "pp", "cp", "tp_size", "pp_size", "cp_size"):
            try:
                model_ranks *= max(1, int(value))
            except ValueError:
                pass
    return max(1, world // model_ranks)


def _timeout() -> timedelta:
    try:
        return timedelta(seconds = max(60, int(os.environ.get(TIMEOUT_ENV, "") or 1800)))
    except ValueError:
        return timedelta(seconds = 1800)


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


_LAST_ERROR: dict = {}  # rank -> its last reported error line, quoted when that rank ends the run


def _pump(
    stream,
    prefix: str,
    r: int = 0,
) -> None:
    try:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace")
            if line.startswith("error: "):
                _LAST_ERROR[r] = line[len("error: ") :].strip()[:500]
            try:
                sys.stderr.write(prefix + line)
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


def wait_for_ranks(timeout: float = 600.0) -> Optional[str]:
    """Rank 0 waits for the other ranks to finish their rank-local saves before the run is reported done.
    None when every rank exited cleanly, else the error to report instead of completion."""
    deadline = time.monotonic() + timeout
    for r, proc in enumerate(_PROCS, start = 1):
        try:
            code = proc.wait(timeout = max(0.1, deadline - time.monotonic()))
        except Exception:  # noqa: BLE001
            _CLOSING.set()
            return f"Multi-GPU training: rank {r} did not finish its save within {int(timeout)} s."
        if code != 0:
            _CLOSING.set()  # reported here; the watcher must not end the process first
            time.sleep(0.5)  # the pump may still be reading the rank's last lines
            reason = _LAST_ERROR.get(r)
            return f"Multi-GPU training: rank {r} exited with code {code} while saving." + (
                f" Rank {r}: {reason}" if reason else ""
            )
    return None


def _group_backend(plan: dict) -> str:
    import torch
    import torch.distributed as dist

    if (
        plan.get("distributed_backend") in ("nccl", "rccl")
        and torch.cuda.is_available()
        and dist.is_nccl_available()
    ):
        return "nccl"  # RCCL on ROCm builds answers to the same name
    return "gloo"


def _jsonable(value):
    return str(value)


def launch_ranks(
    config: dict,
    event_queue: Any,
    *,
    entry: tuple = _DEFAULT_ENTRY,
    python: Optional[str] = None,
) -> int:
    plan = config.get("parallel") or {}
    world = int(plan.get("world_size") or 1)
    if world <= 1 or plan.get("launcher") != "multi_process":
        return 1
    if os.environ.get(CHILD_ENV):
        return world
    module, name = entry
    if not _MODULE_NAME.match(module) or not name.isidentifier():
        raise ValueError(f"launch_ranks: bad entry {entry!r}")
    for key in _LAYOUT_ENV:
        os.environ.pop(key, None)
    if plan.get("spec"):
        os.environ["UNSLOTH_PARALLEL"] = str(plan["spec"])
    for key, value in (plan.get("env") or {}).items():
        # Only the layout knobs the planner sets; one the user exported for the server wins.
        if key in _PLAN_ENV:
            os.environ.setdefault(key, str(value))
    # All ranks are local: bootstrap on loopback unless the user set an interface.
    if sys.platform.startswith("linux"):
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    if not config.get("output_dir") and not config.get("resume_from_checkpoint"):
        # The run dir is named from the clock; every rank must use rank 0's.
        from utils.training_runs import build_default_output_dir_name
        config["output_dir"] = build_default_output_dir_name(
            config["model_name"], config.get("project_name")
        )
    try:
        # Clear once here: per-rank clears would delete modules a faster rank just wrote.
        from utils.cache_cleanup import clear_unsloth_compiled_cache
        clear_unsloth_compiled_cache()
        os.environ[CACHE_CLEARED_ENV] = "1"
    except Exception:  # noqa: BLE001 -- the load clears it as before
        pass
    from core.training.diffusion_ddp import _loopback_store

    store, port = _loopback_store(world)
    os.environ.update(
        MASTER_ADDR = _HOST,
        MASTER_PORT = str(port),
        **{_WORLD_ENV: str(world)},
    )
    os.environ.update(_rank_env(os.environ, 0, world))
    os.environ.pop(CHILD_ENV, None)
    backend = _group_backend(plan)
    base = dict(os.environ)
    base["PYTHONPATH"] = os.pathsep.join([_backend_dir(), *filter(None, [base.get("PYTHONPATH")])])
    base[_PARENT_ENV] = str(os.getpid())
    base["PYTHONUNBUFFERED"] = "1"
    child_config = dict(config)
    # Children inherit rank 0's narrowed mask; re-applying ids would narrow twice.
    child_config["resolved_gpu_ids"] = None
    try:
        from utils.account_context import current_account
        account = dataclasses.asdict(
            current_account()
        )  # a ContextVar: a new interpreter starts as the owner
    except Exception:  # noqa: BLE001
        account = None
    payload = json.dumps(
        {"config": child_config, "account": account, "port": port, "backend": backend},
        default = _jsonable,
    ).encode("utf-8")
    code = (
        "from core.training.parallel_launch import rank_main; "
        f"from {module} import {name} as _entry; rank_main(_entry)"
    )
    atexit.register(_terminate_children)
    try:
        for r in range(1, world):
            env = _rank_env(base, r, world)
            env[CHILD_ENV] = str(r)
            proc = subprocess.Popen(
                [python or sys.executable, "-c", code],
                stdin = subprocess.PIPE,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                env = env,
                cwd = os.getcwd(),
            )
            _PROCS.append(proc)
            threading.Thread(
                target = _pump, args = (proc.stdout, f"[rank {r}] ", r), daemon = True
            ).start()
            proc.stdin.write(payload)
            proc.stdin.close()
    except BaseException:
        _terminate_children()
        raise

    def _watch():
        while not _CLOSING.is_set():
            for r, proc in enumerate(_PROCS, start = 1):
                exit_code = proc.poll()
                if exit_code not in (None, 0) and not _CLOSING.is_set():
                    time.sleep(0.5)  # the pump may still be reading the rank's last lines
                    reason = _LAST_ERROR.get(r)
                    message = (
                        f"Multi-GPU training: rank {r} of {world} exited with code {exit_code}; stopping the run. "
                        + (
                            f"Rank {r}: {reason}"
                            if reason
                            else f"The server log has that rank's error (lines starting with [rank {r}])."
                        )
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
    import torch.distributed as dist

    dist.init_process_group(backend, store = store, rank = 0, world_size = world, timeout = _timeout())
    return world


class _RankEvents:
    def __init__(self, r: int):
        self.rank = r
        # A handled error must still exit non-zero so rank 0 does not wait in a collective.
        self.failed = False

    def put(self, event: Any, *args, **kwargs) -> None:
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind == "error":
            self.failed = True
        if kind in ("error", "warning"):
            text = event.get("error") or event.get("message") or ""
            print(f"{kind}: {text}", file = sys.stderr, flush = True)
            if kind == "error" and event.get("stack"):
                print(event["stack"], file = sys.stderr, flush = True)

    def put_nowait(self, event: Any) -> None:
        self.put(event)


class _NoStop:
    """Never yields; rank 0's broadcast stops ranks 1..N-1."""

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


def rank_main(target: Callable[..., Any]) -> None:
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
    payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    code = 0
    sink = _RankEvents(r)
    try:
        import torch.distributed as dist

        store = dist.TCPStore(
            _HOST, int(payload["port"]), world_size(), is_master = False, timeout = _timeout()
        )
        dist.init_process_group(
            payload["backend"], store = store, rank = r, world_size = world_size(), timeout = _timeout()
        )
        kwargs = {"event_queue": sink, "stop_queue": _NoStop(), "config": payload["config"]}
        account = payload.get("account")
        if account:
            from core.training.account_jobs import run_account_child
            from utils.account_context import AccountContext
            run_account_child(
                account = AccountContext(**account),
                job_module = target.__module__,
                job_target = target.__name__,
                **kwargs,
            )
        else:
            target(**kwargs)
    except BaseException:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        code = 1
    if sink.failed:
        code = code or 2
    sys.stdout.flush()
    sys.stderr.flush()
    # No communicator teardown: NCCL's finalize waits on peers that may already be gone (rank 0 may still be saving).
    os._exit(code)


def sync_stop(should_stop: bool, save_on_stop: bool) -> tuple:
    """Broadcast rank 0's (stop, save). Every rank must call it at the same point (it is a collective)."""
    if not is_multi_process():
        return should_stop, save_on_stop
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
    try:
        dist.broadcast(flags, src = 0)
    except Exception:
        # A dead peer's socket closes before its exit shows: let the watcher name the rank.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not any(p.poll() not in (None, 0) for p in _PROCS):
            time.sleep(0.1)
        if any(p.poll() not in (None, 0) for p in _PROCS):
            time.sleep(5)
        raise
    values = flags.tolist()
    return bool(values[0]), bool(values[1])
