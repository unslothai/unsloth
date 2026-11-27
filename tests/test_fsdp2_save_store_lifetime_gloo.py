# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The FSDP2 writer handshakes (fsdp2_save._wait_for_writer, _share_from_writer) on a gloo CPU world
of 2 with a TCPStore, which rank 0 hosts. Rank 0 writes, returns, destroys the process group and
exits at once, as the end of a training script does; rank 1 is slowed down before or inside its store
reads. Rank 1 must still read rank 0's outcome: rank 0's store has to outlive it.
fsdp2_save.py is loaded on its own (no unsloth import), so a rank starts in about a second."""

import datetime
import importlib.util
import os
import pathlib
import sys
import time
import types

import pytest

if __import__("multiprocessing").parent_process() is not None or getattr(
    __import__("multiprocessing").current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

MODELS = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models"
DELAY_S = 1.5


def _load_fsdp2_save():
    package = types.ModuleType("_fsdp2_save_pkg")
    package.__path__ = [str(MODELS)]
    sys.modules["_fsdp2_save_pkg"] = package
    spec = importlib.util.spec_from_file_location(
        "_fsdp2_save_pkg.fsdp2_save", MODELS / "fsdp2_save.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _SlowGetStore:
    """Rank 1's view of the store: each `get` (after its `wait` returned) first sleeps."""

    def __init__(self, store):
        self._store = store

    def get(self, key):
        time.sleep(DELAY_S)
        return self._store.get(key)

    def __getattr__(self, name):
        return getattr(self._store, name)


def _worker(rank, world, port, case, out):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method = f"tcp://127.0.0.1:{port}",
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = 60),  # a cold runner can take long to start rank 0
    )
    save = _load_fsdp2_save()
    writer = rank == 0
    result = None
    try:
        if case in ("slow_get", "writer_failed") and not writer:
            store = _SlowGetStore(dist.distributed_c10d._get_default_store())
            dist.distributed_c10d._get_default_store = lambda: store
        if case == "reader_never_comes":
            # A rank that skips a handshake (a caller bug) costs rank 0 only a bounded exit wait.
            os.environ["UNSLOTH_FSDP2_SAVE_EXIT_WAIT"] = "1"
            if not writer:
                out.put((rank, {"result": "skipped"}))
                return
        if case == "late_reader" and not writer:
            time.sleep(DELAY_S)  # still finishing its part of the gather
        if case == "export_result":
            # fsdp2_export_on_writer's order: the writer publishes its result inside the gather, the
            # others read it after the gather's own handshake.
            if writer:
                save._share_from_writer(dist, True, 0, "result", ("ok", 42))
                save._wait_for_writer(dist, True, 0)
            else:
                save._wait_for_writer(dist, False, 0)
                time.sleep(DELAY_S)
                result = save._share_from_writer(dist, False, 0, "result")
        elif case == "writer_failed":
            failure = OSError("disk full") if writer else None
            try:
                save._wait_for_writer(dist, writer, 0, failed = failure)
            except save.WriterFailed as exc:
                result = str(exc)
        else:
            save._wait_for_writer(dist, writer, 0)
            result = "done"
        out.put((rank, {"result": result}))
    except Exception as exc:
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}"}))
    finally:
        # No barrier: rank 0 tears down its store the moment it is done, as a script's end does.
        try:
            dist.destroy_process_group()
        except Exception:
            pass


@pytest.mark.skipif(
    sys.platform == "win32",
    reason = "the gloo save tests run on Linux and macOS (as test_fsdp2_save_gloo)",
)
@pytest.mark.parametrize(
    "case", ["slow_get", "late_reader", "export_result", "writer_failed", "reader_never_comes"]
)
def test_rank0_store_outlives_the_readers(case):
    import torch.multiprocessing as mp

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _rendezvous_port import free_port

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = mp.start_processes(
        _worker, args = (2, free_port(), case, out), nprocs = 2, join = False, start_method = "spawn"
    )
    start = time.time()
    deadline = start + 150
    while not procs.join(timeout = 1):
        if time.time() > deadline:
            for p in procs.processes:
                p.kill()
            pytest.fail("the ranks did not finish within 150 s")
    results = {}
    while len(results) < 2:
        rank, r = out.get(timeout = 5)
        results[rank] = r
    for rank, r in results.items():
        assert "crash" not in r, (rank, r["crash"])
    if case == "reader_never_comes":
        assert (
            time.time() - start < 120
        ), "rank 0 waited the default 300 s, not UNSLOTH_FSDP2_SAVE_EXIT_WAIT"
        assert results[0]["result"] == "done", results
    elif case == "export_result":
        assert tuple(results[1]["result"]) == ("ok", 42), results
    elif case == "writer_failed":
        assert "disk full" in results[1]["result"], results
    else:
        assert results[1]["result"] == "done", results


class _CountingStore:
    """`add` only, as the guard uses it; the readers have counted once `released` is set."""

    def __init__(self, world):
        self.world = world
        self.released = False

    def add(self, key, amount):
        return self.world if self.released else 1


def test_a_new_process_group_gets_its_own_guard(monkeypatch):
    """A guard still running for a destroyed group's store must not stand in for the new group's."""
    save = _load_fsdp2_save()
    monkeypatch.setenv("UNSLOTH_FSDP2_SAVE_EXIT_WAIT", "30")
    dist = types.SimpleNamespace(get_rank = lambda: 0, get_world_size = lambda: 2)
    old, new = _CountingStore(2), _CountingStore(2)
    try:
        save._handshake_done(dist, old)
        old_guard = save._STORE_GUARD["thread"]
        assert old_guard is not None and old_guard.is_alive()
        save._handshake_done(dist, new)
        new_guard = save._STORE_GUARD["thread"]
        assert new_guard is not None and new_guard is not old_guard and new_guard.is_alive()
    finally:
        old.released = new.released = True
        for thread in (save._STORE_GUARD["thread"], locals().get("old_guard")):
            if thread is not None:
                thread.join(timeout = 5)
