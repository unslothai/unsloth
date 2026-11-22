# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Block CUDA graphs auto mode under DDP (gloo, world 2, CPU). By default a process group keeps graphs
off and no collective is added. With UNSLOTH_BLOCK_CUDAGRAPHS_DDP=1 every rank probes alone, then the
ranks agree with one all_reduce in a grad-synced DDP forward (the hook is attached when the DDP wrapper
is built, so a rank-local first forward cannot shift it), so graphs go on in the same forward on every
rank or on none. A veto on one rank (GPU-bound, a shape change, a block outside the DDP module,
context parallel, a runtime fallback after graphs went on) turns graphs off everywhere; DDP.join() (no vote and no count
inside the window, so uneven inputs leave the schedule aligned), a rank that never finishes probing and
a probe started before the process group keep them off. Outside a
DDP forward a process group still keeps graphs off. Compile, timing and the CUDA checks are stubbed as
in test_block_cudagraphs_auto.py."""

import multiprocessing
import os
import sys

import pytest

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import contextlib  # noqa: E402
import importlib.util  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not dist.is_available() or not dist.is_gloo_available(),
    reason = "two gloo ranks over TCP localhost",
)

_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth", "models"
)
_STEPS = 8


def _load(name = "_ublock_compile_cg_auto_ddp"):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_DIR, "_block_compile.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _free_port():
    from tests._rendezvous_port import free_port
    return free_port()


class _LooksCuda:
    is_cuda = True


class _Stack(torch.nn.Module):
    def __init__(self, layer_cls):
        super().__init__()
        torch.manual_seed(0)
        self.layers = torch.nn.ModuleList([layer_cls(), layer_cls()])
        self.__dict__["extra"] = None  # a block outside the module tree, not a submodule

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        if self.__dict__["extra"] is not None:
            h = self.__dict__["extra"](h)
        return h


def _scenario(
    bc,
    rank,
    name,
    steps = _STEPS,
    host = 3e-3,
    gpu = 1e-3,
    lengths = None,
    ddp = True,
    sync_every = 1,
    before = None,
    join = False,
    single_steps = 0,
    prelude = None,
    windows = None,
):
    """One fresh auto run; returns the block call tags per step and the final auto phase / why.
    prelude(model, stack) runs once after the DDP wrapper is built; windows replaces the plain loop
    with (context factory, steps) pairs and records votes and the session schedule around each."""
    bc._CUDAGRAPH_STATE = bc._new_cudagraph_state()
    calls = []
    releases = []
    votes = [0]
    reduces = [0]

    def release_spy():
        releases.append(True)
        bc._CUDAGRAPH_STATE["release"] = "done"

    bc._release_graph_pool = release_spy

    def all_reduce_spy(*args, **kwargs):
        # Every all_reduce (DDP.join() issues its own); votes = the ones Unsloth's module issues.
        reduces[0] += 1
        if sys._getframe(1).f_globals.get("__name__") == bc.__name__:
            votes[0] += 1
        return _ALL_REDUCE(*args, **kwargs)

    dist.all_reduce = all_reduce_spy

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
    single = []
    if single_steps:
        # The same model trained in one process before the process group is visible.
        real = dist.is_initialized
        dist.is_initialized = lambda: False
        try:
            for _ in range(single_steps):
                calls.clear()
                stack(torch.randn(2, 16, 8, requires_grad = True)).sum().backward()
                single.append(list(calls))
        finally:
            dist.is_initialized = real
    model = torch.nn.parallel.DistributedDataParallel(stack) if ddp else stack
    votes[0] = reduces[0] = 0
    if prelude is not None:
        prelude(model, stack)
    tags = []

    def step(i):
        if before is not None:
            before(i, stack, AutoDecoderLayer)
        calls.clear()
        length = lengths[i] if lengths else 16
        x = torch.randn(2, length, 8, requires_grad = True)
        accumulate = ddp and sync_every > 1 and i % sync_every != sync_every - 1
        with model.no_sync() if accumulate else contextlib.nullcontext():
            model(x).sum().backward()
        tags.append(list(calls))

    marks = []

    def mark():
        session = bc._CUDAGRAPH_STATE["auto"]["ddp"]
        schedule = (session["count"], session["next"], session["done"]) if session else None
        marks.append({"votes": votes[0], "schedule": schedule, "steps": len(tags)})

    if windows is None:
        with model.join() if join else contextlib.nullcontext():
            for i in range(steps):
                step(i)
    else:
        i = 0
        for context, n in windows:
            mark()
            with context(model):
                for _ in range(n):
                    step(i)
                    i += 1
        mark()
    auto = bc._CUDAGRAPH_STATE["auto"]
    return {
        "single": single,
        "steps": tags,
        "phase": auto["phase"],
        "why": auto["why"],
        "off": bc._CUDAGRAPH_STATE["off"],
        "votes": votes[0],
        "all_reduces": reduces[0],
        "releases": len(releases),
        "marks": marks,
        "hooks": len(stack._forward_pre_hooks)
        + (len(model._forward_pre_hooks) if model is not stack else 0),
    }


def _graphs_from(s):
    return next((i for i, step in enumerate(s["steps"]) if "graphs" in step), None)


_ALL_REDUCE = dist.all_reduce


def _stub(bc):
    bc._CUDAGRAPH_MODE = "auto"
    # CPU tensors: every static rule runs except "not a CUDA tensor" and the platform gate (torch < 2.13
    # or a ROCm build would turn graphs off before the DDP agreement this test is about).
    bc._auto_platform_reason = lambda: None
    bc._auto_is_cuda = lambda hs: True
    static = bc._auto_static_reason
    bc._auto_static_reason = lambda layer, hs: static(layer, _LooksCuda())
    bc._AUTO_DDP_RECHECK = 1
    return bc


def _run_default_off(rank, res):
    # Without UNSLOTH_BLOCK_CUDAGRAPHS_DDP=1: main's behaviour, no hook and no collective of ours.
    bc = _stub(_load("_ublock_compile_cg_auto_ddp_default"))
    res["default_off"] = _scenario(bc, rank, "default_off")
    res["default_off"]["init_patched"] = bool(
        getattr(
            torch.nn.parallel.DistributedDataParallel.__init__, "_unsloth_block_ddp_enroll", False
        )
    )


def _run_eval_first(rank, res, bc):
    # Rank 0 alone runs an eval forward through the DDP wrapper before training (no buffers, so DDP
    # itself issues no collective there); its first wrapped block call comes a forward early.
    def prelude(model, stack):
        if rank == 0:
            model.eval()
            with torch.no_grad():
                model(torch.randn(2, 16, 8))
            model.train()

    res["eval_first"] = _scenario(bc, rank, "eval_first", prelude = prelude)


def _run_nosync_first(rank, res, bc):
    # Rank 0 alone runs one extra no_sync() microbatch before training.
    def prelude(model, stack):
        if rank == 0:
            with model.no_sync():
                model(torch.randn(2, 16, 8, requires_grad = True)).sum().backward()

    res["nosync_first"] = _scenario(bc, rank, "nosync_first", prelude = prelude)


def _run_join_active(rank, res, bc):
    # Production cadence: graphs on, then an uneven DDP.join() window (rank 0: 3 inputs, rank 1: none),
    # then synced training again with join off.
    bc._AUTO_DDP_RECHECK = 64
    res["join_active"] = _scenario(
        bc,
        rank,
        "join_active",
        steps = 12,
        windows = [
            (lambda model: contextlib.nullcontext(), 6),
            (lambda model: model.join(), 3 if rank == 0 else 0),
            (lambda model: model.join(enable = False), 6),
        ],
    )
    bc._AUTO_DDP_RECHECK = 1


def _worker(
    rank,
    world,
    port,
    out,
    group = "main",
):
    # A CI shard may run with UNSLOTH_COMPILE_DISABLE=1 or graphs forced on: every rank starts from
    # the defaults this test is about, before the module reads them at import.
    for key in [k for k in os.environ if k.startswith("UNSLOTH_")]:
        os.environ.pop(key)
    os.environ["UNSLOTH_BLOCK_CUDAGRAPHS"] = "auto"
    os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = str(port))
    import datetime

    # A desynchronised collective fails at this timeout instead of hanging the test.
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 20)
    )
    try:
        res = {}
        if group == "main":
            _run_default_off(rank, res)
        os.environ["UNSLOTH_BLOCK_CUDAGRAPHS_DDP"] = "1"
        bc = _stub(_load())
        if group == "eval_first":
            _run_eval_first(rank, res, bc)
        elif group == "nosync_first":
            _run_nosync_first(rank, res, bc)
        elif group == "join_active":
            _run_join_active(rank, res, bc)
        else:
            _run_main(rank, res, bc)
        dist.barrier()
        out.put({"rank": rank, **res})
    except Exception as e:  # pragma: no cover - reported to the parent
        import traceback
        out.put({"rank": rank, "error": f"{e!r}\n{traceback.format_exc()}"})
    finally:
        dist.destroy_process_group()


def _run_main(rank, res, bc):
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

    # Rank 1 is ready two forward passes later and runs behind inside each no_sync() window.
    def lag(i, stack, layer_cls):
        if rank == 1:
            time.sleep(0.15)

    warmup = bc._AUTO_WARMUP_PASSES
    bc._AUTO_WARMUP_PASSES = warmup if rank == 0 else warmup + 2
    res["skewed_ready"] = _scenario(bc, rank, "skewed_ready", steps = 12, sync_every = 4, before = lag)
    bc._AUTO_WARMUP_PASSES = warmup

    # After graphs are on, rank 1 alone falls back (a new batch shape over the cap).
    cap = bc._CUDAGRAPH_MAX_SHAPES
    bc._CUDAGRAPH_MAX_SHAPES = 1
    res["late_veto"] = _scenario(
        bc,
        rank,
        "late_veto",
        steps = 12,
        lengths = [16] * 12 if rank == 0 else [16] * 6 + [24] * 6,
    )
    bc._CUDAGRAPH_MAX_SHAPES = cap

    # After graphs are on, rank 1's forward also calls a block outside the DDP module.
    def outside(i, stack, layer_cls):
        if rank == 1 and i == 6:
            stack.__dict__["extra"] = layer_cls().train()

    res["late_outside_block"] = _scenario(bc, rank, "late_outside_block", steps = 12, before = outside)

    # After graphs are on, rank 1 enters context parallel.
    def cp(i, stack, layer_cls):
        if rank == 1 and i == 6:
            sys.modules["unsloth.context_parallel"] = types.SimpleNamespace(
                get_cp_manager = lambda: object()
            )

    res["late_context_parallel"] = _scenario(bc, rank, "late_context_parallel", steps = 12, before = cp)
    sys.modules.pop("unsloth.context_parallel", None)

    # Rank 0 probed and turned graphs on in one process before the process group was visible.
    res["pg_after_probe"] = _scenario(
        bc, rank, "pg_after_probe", single_steps = 6 if rank == 0 else 0
    )

    # Uneven inputs under DDP.join(): no vote can be taken, so graphs stay off and nothing hangs.
    res["join"] = _scenario(bc, rank, "join", steps = 8 if rank == 0 else 6, join = True)

    # Rank 1 never finishes probing: the bounded probe turns graphs off on both.
    probes = bc._AUTO_DDP_MAX_PROBES if hasattr(bc, "_AUTO_DDP_MAX_PROBES") else None
    bc._AUTO_DDP_MAX_PROBES = 5
    bc._AUTO_WARMUP_PASSES = warmup if rank == 0 else 10**6
    res["never_ready"] = _scenario(bc, rank, "never_ready")
    bc._AUTO_WARMUP_PASSES = warmup
    if probes is not None:
        bc._AUTO_DDP_MAX_PROBES = probes


def _spawn(group):
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(
        _worker, args = (2, _free_port(), out, group), nprocs = 2, join = True, start_method = "spawn"
    )
    results = {}
    for _ in range(2):
        r = out.get(timeout = 30)
        results[r["rank"]] = r
    return results


_RESULTS = {}


def _results(group = "main"):
    if group not in _RESULTS:
        _RESULTS[group] = _spawn(group)
    results = _RESULTS[group]
    errors = {rank: r["error"] for rank, r in results.items() if "error" in r}
    assert not errors, errors
    assert sorted(results) == [0, 1]
    return results


def _all_plain(s):
    return all("graphs" not in step for step in s["steps"])


def test_default_ddp_keeps_graphs_off_without_any_collective():
    for r in _results().values():
        s = r["default_off"]
        assert s["phase"] == "off" and s["why"] == "distributed", s
        assert all(step == ["plain", "plain"] for step in s["steps"]), s
        assert s["all_reduces"] == 0 and s["hooks"] == 0 and not s["init_patched"], s


@pytest.mark.parametrize("group", ["eval_first", "nosync_first"])
def test_rank_local_first_forward_does_not_shift_enrollment(group):
    r = _results(group)
    s0, s1 = r[0][group], r[1][group]
    for s in (s0, s1):
        assert s["phase"] == "on" and s["off"] is None, s
        assert _graphs_from(s) is not None, s
    assert s0["steps"] == s1["steps"], (s0["steps"], s1["steps"])
    assert s0["votes"] == s1["votes"] > 0, (s0, s1)


def test_uneven_join_window_keeps_the_vote_schedule_aligned():
    r = _results("join_active")
    s0, s1 = r[0]["join_active"], r[1]["join_active"]
    for s in (s0, s1):
        _, before, inside, after = s["marks"]  # start, join entered, join left, end
        assert _graphs_from(s) is not None and _graphs_from(s) < 6, s  # on before the window
        assert before["schedule"] is not None and not before["schedule"][2], s
        # No vote and no count inside the window, on the rank with inputs and the one without.
        assert inside["schedule"] == before["schedule"], s
        assert inside["votes"] == before["votes"], s
        # Synced training resumes on one schedule: a vote is taken and graphs end off everywhere.
        assert after["votes"] > inside["votes"], s
        assert s["off"] is not None, s
        assert all("graphs" not in step for step in s["steps"][-3:]), s
    # Rank 0's join forwards ran plain.
    assert all("graphs" not in step for step in s0["steps"][6:9]), s0
    assert s0["marks"][2]["schedule"] == s1["marks"][2]["schedule"], (s0, s1)
    assert s0["marks"][3]["schedule"] == s1["marks"][3]["schedule"], (s0, s1)
    # Rank 1 had no input in the window: rank 0's join veto reaches it at the first vote after it.
    assert s0["why"] == "DDP join (uneven inputs)", s0
    assert s1["why"] == "rank 0: graphs off there", s1


def test_process_group_without_ddp_forward_keeps_graphs_off():
    for r in _results().values():
        s = r["no_ddp"]
        assert s["phase"] == "off" and s["why"] == "distributed, not DDP", s
        assert all(step == ["plain", "plain"] for step in s["steps"]), s


def test_ddp_ranks_turn_graphs_on_together():
    r = _results()
    s0, s1 = r[0]["agree_on"], r[1]["agree_on"]
    for s in (s0, s1):
        assert s["phase"] == "on" and s["why"] is None and s["off"] is None, s
        first = _graphs_from(s)
        assert first is not None, s
        assert all(step == ["graphs", "graphs"] for step in s["steps"][first:]), s
    assert s0["steps"] == s1["steps"], (s0, s1)


def test_ranks_ready_at_different_passes_switch_in_the_same_forward():
    r = _results()
    s0, s1 = r[0]["skewed_ready"], r[1]["skewed_ready"]
    assert _graphs_from(s0) is not None, s0
    # Same tags on every step: no forward where one rank replays graphs and the other runs plain.
    assert s0["steps"] == s1["steps"], (s0["steps"], s1["steps"])
    # The switch happens on a synced forward (every 4th), the only place the ranks vote.
    assert _graphs_from(s0) % 4 == 3, s0


def test_one_gpu_bound_rank_keeps_graphs_off_everywhere():
    r = _results()
    s0, s1 = r[0]["one_gpu_bound"], r[1]["one_gpu_bound"]
    assert s1["phase"] == "off" and s1["why"].startswith("GPU-bound"), s1
    assert s0["phase"] == "off" and s0["why"].startswith("rank 1"), s0
    for s in (s0, s1):
        assert _all_plain(s), s


def test_shape_change_on_one_rank_keeps_graphs_off_everywhere():
    r = _results()
    s0, s1 = r[0]["one_shape_change"], r[1]["one_shape_change"]
    assert s1["phase"] == "off" and s1["why"] == "batch shape changed", s1
    assert s0["phase"] == "off" and s0["why"].startswith("rank 1"), s0
    for s in (s0, s1):
        assert _all_plain(s), s


@pytest.mark.parametrize("name", ["late_veto", "late_outside_block", "late_context_parallel"])
def test_late_veto_on_one_rank_turns_graphs_off_everywhere(name):
    r = _results()
    s0, s1 = r[0][name], r[1][name]
    for s in (s0, s1):
        first = _graphs_from(s)
        assert first is not None and first < 6, s  # graphs were on before rank 1's veto
        assert s["off"] is not None, s
        assert all("graphs" not in step for step in s["steps"][-3:]), s
        assert s["releases"] >= 1, s  # the graph pool is released on the vetoing rank too


def test_probe_before_the_process_group_does_not_turn_graphs_on_under_ddp():
    r = _results()
    assert any("graphs" in step for step in r[0]["pg_after_probe"]["single"]), r[0][
        "pg_after_probe"
    ]
    for rank in (0, 1):
        s = r[rank]["pg_after_probe"]
        assert _all_plain(s), (rank, s)


def test_ddp_join_keeps_graphs_off_without_hanging():
    for r in _results().values():
        s = r["join"]
        assert s["why"] == "DDP join (uneven inputs)", s
        assert _all_plain(s), s


def test_rank_that_never_finishes_probing_keeps_graphs_off_everywhere():
    for r in _results().values():
        s = r["never_ready"]
        assert s["phase"] == "off", s
        assert "still probing" in (s["why"] or ""), s
        assert _all_plain(s), s
