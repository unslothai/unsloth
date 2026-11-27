# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""native_save_optimizer_sharded called directly on a gloo CPU world of 2 with a TCPStore, which rank
0 hosts. Rank 0 is the only node leader, so it clears and commits the checkpoint directory in
fsdp2_save._writers_in_turn; rank 1 only reads the outcome. Rank 0 returns, destroys its process
group and exits at once, as a script ending on that save does, and rank 1 reads the commit's outcome
only once rank 0's process is gone (or after EXIT_WAIT_S). Rank 1 must still read it: rank 0's
store has to outlive it."""

import datetime
import os
import sys
import time

import pytest
import torch

if __import__("multiprocessing").parent_process() is not None or getattr(
    __import__("multiprocessing").current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_guards_gloo import _tiny_llama  # noqa: E402

EXIT_WAIT_S = 10.0


class _LateCommitReadStore:
    """Rank 1's view of the store: its read of the commit's outcome waits until rank 0 has returned
    from the save and its process is gone (at most EXIT_WAIT_S; with the store guard rank 0 stays)."""

    def __init__(self, store, returned, pid_file):
        self._store = store
        self._returned = returned
        self._pid_file = pid_file

    def get(self, key):
        if "dcp_commit" in key:
            while not os.path.exists(self._returned):
                time.sleep(0.01)
            pid = int(open(self._pid_file, encoding = "utf-8").read())
            deadline = time.time() + EXIT_WAIT_S
            while time.time() < deadline and _alive(pid):
                time.sleep(0.05)
        return self._store.get(key)

    def __getattr__(self, name):
        return getattr(self._store, name)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _worker(rank, world, port, directory, out):
    returned, pid_file = directory + ".returned", directory + ".pid0"
    os.environ.update(RANK = str(rank), WORLD_SIZE = str(world), LOCAL_RANK = str(rank))
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method = f"tcp://127.0.0.1:{port}",
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = 60),  # a cold runner can take long to start rank 0
    )
    try:
        native = _load_native()
        model = native.shard_model(_tiny_llama(), native.FSDP2Config(param_dtype = torch.float32))
        opt = torch.optim.AdamW(model.parameters(), lr = 1e-3, foreach = False)
        model(torch.tensor([[1, 2, 3]])).logits.float().mean().backward()
        opt.step()
        if rank == 0:
            with open(pid_file, "w", encoding = "utf-8") as f:
                f.write(str(os.getpid()))
        dist.barrier()
        if rank == 1:
            store = _LateCommitReadStore(
                dist.distributed_c10d._get_default_store(), returned, pid_file
            )
            dist.distributed_c10d._get_default_store = lambda: store
        native.native_save_optimizer_sharded(model, opt, directory, step = 1)
        if rank == 0:
            open(returned, "w").close()
        out.put((rank, {"committed": os.path.isdir(directory)}))
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
def test_rank0_store_outlives_the_sharded_optimizer_commit(tmp_path):
    pytest.importorskip("transformers")
    import torch.multiprocessing as mp

    from _rendezvous_port import free_port

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    args = (2, free_port(), str(tmp_path / "optimizer_dcp"), out)
    procs = mp.start_processes(_worker, args = args, nprocs = 2, join = False, start_method = "spawn")
    deadline = time.time() + 180
    while not procs.join(timeout = 1):
        if time.time() > deadline:
            for p in procs.processes:
                p.kill()
            pytest.fail("the ranks did not finish within 180 s")
    results = {}
    while len(results) < 2:
        rank, r = out.get(timeout = 5)
        results[rank] = r
    for rank, r in results.items():
        assert "crash" not in r, (rank, r["crash"])
        assert r["committed"], (rank, results)
