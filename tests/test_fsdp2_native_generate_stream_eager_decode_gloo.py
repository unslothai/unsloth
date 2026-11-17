# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 (gloo CPU world of 2): a block-streamed `generate` decodes without CUDA graphs.

Each streamed decode step all-gathers every decoder block, so a captured decode graph holds NCCL
collectives. NCCL keeps a communicator with graph-captured work alive until the graph is destroyed,
and the generic FastModel path keeps its decode graph for the next call: on 2 GPUs
`destroy_process_group()` after a streamed generate blocked until the NCCL watchdog aborted the
process. The NCCL hang itself needs 2 GPUs; here every decode step records whether decode graphs
would be taken (`_decode_graphs.enabled()`, forced on with UNSLOTH_DECODE_CUDA_GRAPHS=1): off while
blocks stream, unchanged for the whole-model gather, and on again after the call."""

import importlib
import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402
from test_fsdp2_native_generate_streams_blocks_gloo import (  # noqa: E402
    NEW_TOKENS,
    _LoopNet,
    _loop_greedy,
)


class _RecordingNet(_LoopNet):
    """Unsloth-shaped decode loop that records, per decode step, whether decode graphs are enabled."""

    def _decode_step(self, ids):
        graphs = sys.modules["unsloth.models._decode_graphs"]
        self.__dict__.setdefault("_graph_flags", []).append(graphs.enabled())
        return super()._decode_step(ids)


def _worker(rank, world, port, out, mode):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
        UNSLOTH_FSDP2_GENERATE_GATHER = mode,
        UNSLOTH_DECODE_CUDA_GRAPHS = "1",
    )
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 60)
    )
    try:
        native = _load_native()
        graphs = importlib.import_module("unsloth.models._decode_graphs")
        model = _RecordingNet()
        native.shard_model(
            model,
            native.FSDP2Config(
                device = "cpu", units = list(model.layers), fit_layout = False, param_dtype = torch.float32
            ),
        )
        res = {"tokens": _loop_greedy(model), "flags": list(model.__dict__.get("_graph_flags", []))}
        res["enabled_after"] = graphs.enabled()
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(mode, timeout = 180):
    import torch.multiprocessing as mp

    out = mp.get_context("spawn").Queue()
    procs = mp.start_processes(
        _worker,
        args = (2, _free_port(), out, mode),
        nprocs = 2,
        join = False,
        start_method = "spawn",
    )
    deadline = time.time() + timeout
    try:
        while not procs.join(timeout = 5):
            if time.time() > deadline:
                for p in procs.processes:
                    p.kill()
                pytest.fail(f"the ranks did not finish within {timeout} s (hung collective)")
    except mp.ProcessExitedException as exc:
        pytest.fail(f"a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for r in results.values():
        assert "crash" not in r, r["crash"]
    return results


def _reference():
    with torch.no_grad():
        return _loop_greedy(_LoopNet())


def test_streamed_generate_decodes_without_cuda_graphs():
    want = _reference()
    for rank, r in _run("block").items():
        assert r["tokens"] == want, (rank, r, want)
        assert len(r["flags"]) == NEW_TOKENS, (rank, r)
        # Every streamed decode step all-gathers its blocks: no step may be captured into a graph.
        assert not any(r["flags"]), (rank, r)
        # Scoped to the call: graphs are allowed again afterwards.
        assert r["enabled_after"] is True, (rank, r)


def test_whole_model_gather_keeps_cuda_graphs():
    want = _reference()
    for rank, r in _run("all").items():
        assert r["tokens"] == want, (rank, r, want)
        # No collective in the decode step: decode graphs stay as configured.
        assert r["flags"] and all(r["flags"]), (rank, r)
