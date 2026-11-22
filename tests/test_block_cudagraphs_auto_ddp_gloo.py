# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Block CUDA graphs auto mode under DDP (gloo, world 2, CPU): auto may turn graphs on inside a plain
DDP forward, and every rank takes the same decision. Each rank probes alone and publishes its verdict
to the process group's store; graphs start only once every rank said "on", any "off" (GPU-bound, a
batch shape change on one rank) keeps them off everywhere, and a rank that never decides times out to
off instead of hanging the others. Outside a DDP forward a process group still keeps graphs off.
Compile, timing and the CUDA checks are stubbed as in test_block_cudagraphs_auto.py."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib.util  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth", "models"
)
_STEPS = 8


def _load():
    spec = importlib.util.spec_from_file_location(
        "_ublock_compile_cg_auto_ddp", os.path.join(_DIR, "_block_compile.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _LooksCuda:
    is_cuda = True


class _Stack(torch.nn.Module):
    def __init__(self, layer_cls):
        super().__init__()
        torch.manual_seed(0)
        self.layers = torch.nn.ModuleList([layer_cls(), layer_cls()])

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return h


def _scenario(
    bc,
    rank,
    name,
    run_steps = True,
    host = 3e-3,
    gpu = 1e-3,
    lengths = None,
    ddp = True,
):
    """One fresh auto round; returns (block call tags per step, final auto phase / why)."""
    bc._CUDAGRAPH_STATE = bc._new_cudagraph_state()
    calls = []

    def fake_compile(fn, **kwargs):
        tag = "graphs" if kwargs.get("options", {}).get("triton.cudagraphs") else "plain"

        def run(*args, **kw):
            calls.append(tag)
            return fn(*args, **kw)

        return run

    torch.compile = fake_compile

    def fake_time(call, device):
        return call(), host, gpu

    bc._auto_time = fake_time
    torch.cuda.mem_get_info = lambda device = None: (40 * 2**30, 80 * 2**30)

    class AutoDecoderLayer(torch.nn.Module):
        """A frozen weight plus a rank-1 adapter: LoRA, so no "full finetuning" rule."""

        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8).requires_grad_(False)
            self.lora_A = torch.nn.Linear(8, 1, bias = False)
            self.lora_B = torch.nn.Linear(1, 8, bias = False)

        def forward(self, hidden_states, **kwargs):
            return self.proj(hidden_states) + self.lora_B(self.lora_A(hidden_states))

    AutoDecoderLayer.__name__ = f"AutoDecoderLayer_{name}"
    bc._wrap(AutoDecoderLayer, dict(bc._COMPILE_OPTIONS))
    stack = _Stack(AutoDecoderLayer).train()
    model = torch.nn.parallel.DistributedDataParallel(stack) if ddp else stack
    steps = []
    for i in range(_STEPS if run_steps else 0):
        calls.clear()
        L = lengths[i] if lengths else 16
        x = torch.randn(2, L, 8, requires_grad = True)
        model(x).sum().backward()
        steps.append(list(calls))
        if name == "timeout":
            time.sleep(0.3)
    auto = bc._CUDAGRAPH_STATE["auto"]
    return {"steps": steps, "phase": auto["phase"], "why": auto["why"]}


def _worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 60)
    )
    try:
        bc = _load()
        bc._CUDAGRAPH_MODE = "auto"
        # CPU tensors: every static rule runs except "not a CUDA tensor"
        static = bc._auto_static_reason
        bc._auto_static_reason = lambda layer, hs: static(layer, _LooksCuda())
        res = {}
        res["no_ddp"] = _scenario(bc, rank, "no_ddp", ddp = False)
        res["agree_on"] = _scenario(bc, rank, "agree_on")
        res["one_gpu_bound"] = _scenario(
            bc,
            rank,
            "one_gpu_bound",
            host = 3e-3 if rank == 0 else 1e-3,
            gpu = 1e-3 if rank == 0 else 2e-3,
        )
        res["one_shape_change"] = _scenario(
            bc,
            rank,
            "one_shape_change",
            lengths = [16] * _STEPS if rank == 0 else [16, 16, 24] + [16] * 5,
        )
        # Rank 1 never reaches a block call: rank 0 must time out to off, not wait for ever.
        bc._AUTO_AGREE_TIMEOUT_S = 0.5
        bc._auto_static_reason = lambda layer, hs: None
        res["timeout"] = _scenario(bc, rank, "timeout", run_steps = rank == 0, ddp = False)
        dist.barrier()
        out.put({"rank": rank, **res})
    except Exception as e:  # pragma: no cover - reported to the parent
        import traceback
        out.put({"rank": rank, "error": f"{e!r}\n{traceback.format_exc()}"})
    finally:
        dist.destroy_process_group()


def _spawn():
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(
        _worker, args = (2, _free_port(), out), nprocs = 2, join = True, start_method = "spawn"
    )
    results = {}
    while not out.empty():
        r = out.get(timeout = 5)
        results[r["rank"]] = r
    return results


_RESULTS = {}


def _results():
    if not _RESULTS:
        _RESULTS.update(_spawn())
    for r in _RESULTS.values():
        assert "error" not in r, r["error"]
    assert sorted(_RESULTS) == [0, 1]
    return _RESULTS


def test_process_group_without_ddp_forward_keeps_graphs_off():
    for r in _results().values():
        s = r["no_ddp"]
        assert s["phase"] == "off" and s["why"] == "distributed, not DDP", s
        assert all(step == ["plain", "plain"] for step in s["steps"]), s


def test_ddp_ranks_turn_graphs_on_together():
    r = _results()
    first_graphs = []
    for rank in (0, 1):
        s = r[rank]["agree_on"]
        assert s["phase"] == "on" and s["why"] is None, s
        first_graphs.append(next(i for i, step in enumerate(s["steps"]) if "graphs" in step))
        assert all(step == ["graphs", "graphs"] for step in s["steps"][first_graphs[-1] :]), s
    assert first_graphs[0] == first_graphs[1], first_graphs


def test_one_gpu_bound_rank_keeps_graphs_off_everywhere():
    r = _results()
    s0, s1 = r[0]["one_gpu_bound"], r[1]["one_gpu_bound"]
    assert s1["phase"] == "off" and s1["why"].startswith("GPU-bound"), s1
    assert s0["phase"] == "off" and s0["why"].startswith("rank 1: GPU-bound"), s0
    for s in (s0, s1):
        assert all(step == ["plain", "plain"] for step in s["steps"]), s


def test_shape_change_on_one_rank_keeps_graphs_off_everywhere():
    r = _results()
    s0, s1 = r[0]["one_shape_change"], r[1]["one_shape_change"]
    assert s1["phase"] == "off" and s1["why"] == "batch shape changed", s1
    assert s0["phase"] == "off" and s0["why"] == "rank 1: batch shape changed", s0
    for s in (s0, s1):
        assert all(step == ["plain", "plain"] for step in s["steps"]), s


def test_rank_that_never_decides_times_out_to_off():
    s0 = _results()[0]["timeout"]
    assert s0["phase"] == "off" and s0["why"] == "ranks did not agree in time", s0
    assert all(step == ["plain", "plain"] for step in s0["steps"]), s0
