# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""checkpoint_skip_layers = "auto" under torchrun, on CPU gloo: which runs take the collective, that
every rank takes it at the same forward, and that DDP training is unchanged by the plan."""

import datetime
import multiprocessing
import os
import socket
import sys
import types

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GIB = 1 << 30


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _load(
    distributed = None,
    tensor_parallel = False,
    pipeline_parallel = False,
):
    """The planner from _utils.py, with its package-relative imports (is_distributed,
    is_tensor_parallel, is_pipeline_parallel) answered by stubs instead of falling back."""
    sys.path.insert(0, HERE)
    import test_gc_skip_auto as T

    ns, _ = T._load()
    ns["__package__"] = "unsloth_s9stub.models"
    package = types.ModuleType("unsloth_s9stub")
    package.__path__ = []
    models = types.ModuleType("unsloth_s9stub.models")
    models.__path__ = []
    loader = types.ModuleType("unsloth_s9stub.models.loader_utils")
    if distributed is None:
        loader.is_distributed = lambda: int(os.environ.get("WORLD_SIZE", "1")) > 1
    else:
        loader.is_distributed = lambda: distributed
    tp = types.ModuleType("unsloth_s9stub.models.tp_native")
    tp.is_tensor_parallel = lambda model: tensor_parallel
    pp = types.ModuleType("unsloth_s9stub.models.pp_native")
    pp.is_pipeline_parallel = lambda model: pipeline_parallel
    for module in (package, models, loader, tp, pp):
        sys.modules[module.__name__] = module
    return T, ns


def _spoof_cuda(ns, usable, memory):
    # Process-local CPU stand-ins for the CUDA calls the planner makes (as T._fake_cuda does).
    torch.cuda.is_available = lambda: True
    torch.Tensor.is_cuda = property(lambda self: True)
    torch.cuda.reset_peak_memory_stats = lambda device = None: None
    torch.cuda.memory_allocated = lambda device = None: memory["allocated"]
    torch.cuda.max_memory_allocated = lambda device = None: memory["allocated"]
    ns["usable_cuda_bytes"] = lambda device: usable


def _count_collectives():
    import torch.distributed as dist

    calls = {"all_reduce": 0}
    real = dist.all_reduce

    def counted(*args, **kwargs):
        calls["all_reduce"] += 1
        return real(*args, **kwargs)

    dist.all_reduce = counted
    return calls


def _fake_scenario(rank, scenario, usable):
    """The fake 8-layer model of test_gc_skip_auto, with rank-local extra forwards."""
    T, ns = _load(
        tensor_parallel = scenario == "tp",
        pipeline_parallel = scenario == "pp",
    )
    memory = {"allocated": 0}
    _spoof_cuda(ns, usable, memory)
    calls = _count_collectives()
    model = T._FakeModel([GIB // 16] * 8, memory)
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    if scenario == "rank0_pretrain_forward" and rank == 0:
        # A grad-enabled training forward only rank 0 runs before step 1 (no backward).
        model(torch.zeros(1, 16, 4))
    T._fake_step(model, rows = 16)
    if scenario == "rank0_eval_and_generate" and rank == 0:
        # Rank-0-only evaluation / sample generation between steps: eval mode, no grad.
        model.eval()
        model(torch.zeros(1, 16, 4))
        model.train()
        with torch.no_grad():
            model(torch.zeros(1, 16, 4))
        with torch.inference_mode():
            model(torch.zeros(1, 16, 4))
    model(torch.zeros(1, 16, 4))
    T._fake_step(model, rows = 16)
    model(torch.zeros(1, 16, 4))
    return {"kept": list(planner.kept), "all_reduce": calls["all_reduce"]}


def _ddp_scenario(rank, scenario, usable):
    """A tiny Qwen3 under DDP (gloo), HF reentrant gradient checkpointing, 4 optimizer steps."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    static_graph, find_unused, accumulation, auto = scenario
    if static_graph == "fsdp2":
        from torch.distributed.device_mesh import init_device_mesh

        # A CPU mesh built before the CUDA stand-ins, which would make it pick cuda / NCCL.
        mesh = init_device_mesh("cpu", (torch.distributed.get_world_size(),))
    T, ns = _load()
    memory = {"allocated": 0}
    model = T._model(layers = 4)
    if auto:
        _spoof_cuda(ns, usable, memory)
        # Each grad-mode layer call "allocates" 1/16 GiB, so the planner measures a cost.
        for layer in model.model.layers:
            layer.register_forward_hook(
                lambda m, a, o: memory.__setitem__(
                    "allocated", memory["allocated"] + (GIB // 16 if torch.is_grad_enabled() else 0)
                )
            )
        ns["skip_checkpointing"](model, "auto")
    planner = getattr(model, "_unsloth_auto_skip", None)
    calls = _count_collectives()
    if static_graph == "fsdp2":
        try:
            from torch.distributed.fsdp import fully_shard
        except ImportError:
            from torch.distributed._composable.fsdp import fully_shard
        for layer in model.model.layers:
            fully_shard(layer, mesh = mesh)
        fully_shard(model, mesh = mesh)
        ddp = model
        ddp.no_sync = _null
    else:
        ddp = DDP(model, static_graph = static_graph, find_unused_parameters = find_unused)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    generator = torch.Generator().manual_seed(100 + rank)
    losses = []
    for step in range(4):
        for micro in range(accumulation):
            ids = torch.randint(0, 128, (2, 32), generator = generator)
            context = ddp.no_sync() if micro < accumulation - 1 else _null()
            with context:
                loss = ddp(input_ids = ids, labels = ids).loss / accumulation
                loss.backward()
            losses.append(float(loss.detach()))
        optimizer.step()
        optimizer.zero_grad(set_to_none = True)
    digest = float(
        sum(
            (p.full_tensor() if hasattr(p, "full_tensor") else p).detach().double().sum()
            for p in model.parameters()
        )
    )
    return {
        "kept": None if planner is None else (None if planner.kept is None else list(planner.kept)),
        "checkpointed": [bool(layer.gradient_checkpointing) for layer in model.model.layers],
        "losses": losses,
        "digest": digest,
        "all_reduce": calls["all_reduce"],
    }


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _worker(rank, world, port, kind, scenario, usable, queue):
    import torch.distributed as dist

    os.environ["WORLD_SIZE"] = str(world)
    os.environ["RANK"] = str(rank)
    try:
        dist.init_process_group(
            "gloo",
            init_method = f"tcp://127.0.0.1:{port}",
            rank = rank,
            world_size = world,
            timeout = datetime.timedelta(seconds = 30),
        )
        run = _fake_scenario if kind == "fake" else _ddp_scenario
        queue.put((rank, run(rank, scenario, usable[rank]), None))
    except Exception:
        import traceback
        queue.put((rank, None, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _launch(kind, scenario, usable):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target = _worker, args = (r, len(usable), port, kind, scenario, usable, queue))
        for r in range(len(usable))
    ]
    for p in procs:
        p.start()
    results = dict()
    try:
        for _ in procs:
            rank, result, error = queue.get(timeout = 240)
            assert error is None, error
            results[rank] = result
    finally:
        for p in procs:
            p.join(timeout = 60)
            if p.is_alive():
                p.kill()
    return results


@pytest.mark.parametrize("scenario", ["rank0_pretrain_forward", "rank0_eval_and_generate"])
def test_rank_local_forwards_do_not_desync_the_collective(scenario):
    # Rank 0 has room for 8 layers, rank 1 for 2; rank-local forwards must not move rank 0's plan.
    results = _launch("fake", scenario, [4 * GIB, 2 * GIB])
    assert results[0]["kept"] == results[1]["kept"] == [3, 7]
    assert results[0]["all_reduce"] == results[1]["all_reduce"] == 1


@pytest.mark.parametrize("scenario", ["tp", "pp"])
def test_tensor_and_pipeline_parallel_stay_checkpointed_without_a_collective(scenario, capfd):
    results = _launch("fake", scenario, [4 * GIB, 4 * GIB])
    for rank in (0, 1):
        assert results[rank]["kept"] == []
        assert results[rank]["all_reduce"] == 0


def test_process_group_missing_under_torchrun_keeps_every_layer(monkeypatch, capsys):
    monkeypatch.setenv("WORLD_SIZE", "2")
    T, ns = _load(distributed = True)
    memory = T._fake_cuda(monkeypatch, ns, usable = 4 * GIB)
    import torch.distributed as dist

    assert not dist.is_initialized()
    monkeypatch.setattr(
        dist, "all_reduce", lambda *a, **k: pytest.fail("collective without a group")
    )
    model = T._FakeModel([GIB // 16] * 8, memory)
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    T._fake_step(model, rows = 16)
    model(torch.zeros(1, 16, 4))
    assert planner.kept == []
    assert "needs the process group before the second step" in capsys.readouterr().out
    monkeypatch.undo()


@pytest.mark.parametrize("world_size_env", [None, "1"])
def test_single_process_plans_alone_without_a_collective(monkeypatch, world_size_env):
    if world_size_env is None:
        monkeypatch.delenv("WORLD_SIZE", raising = False)
    else:
        monkeypatch.setenv("WORLD_SIZE", world_size_env)
    T, ns = _load()
    memory = T._fake_cuda(monkeypatch, ns, usable = 4 * GIB)
    import torch.distributed as dist

    monkeypatch.setattr(
        dist, "all_reduce", lambda *a, **k: pytest.fail("collective on one process")
    )
    model = T._FakeModel([GIB // 16] * 8, memory)
    ns["skip_checkpointing"](model, "auto")
    planner = model._unsloth_auto_skip
    T._fake_step(model, rows = 16)
    model(torch.zeros(1, 16, 4))
    # (4 GiB - 0.5 GiB step) * 0.75 - 1 GiB = 1.6 GiB: all 8 sixteenth-GiB layers, as before.
    assert planner.kept == list(range(8))
    monkeypatch.undo()


def test_world_size_one_process_group_plans_alone():
    results = _launch("fake", "agree", [4 * GIB])
    assert results[0]["kept"] == list(range(8))
    assert results[0]["all_reduce"] == 0


def _assert_same_training(auto, reference):
    # Skipping the recompute may change fp32 rounding by an ulp on some torch versions (2.13 CPU).
    torch.testing.assert_close(auto["losses"], reference["losses"], rtol = 1e-6, atol = 1e-6)
    torch.testing.assert_close(auto["digest"], reference["digest"], rtol = 1e-6, atol = 1e-6)


@pytest.mark.parametrize(
    "static_graph, find_unused, accumulation",
    # static_graph + no_sync and find_unused_parameters fail with reentrant checkpointing regardless (DDP limits).
    [(False, False, 1), (False, False, 2), (True, False, 1)],
)
def test_ddp_training_unchanged_and_ranks_agree(static_graph, find_unused, accumulation):
    # Rank 0 fits 4, rank 1 fits 2: both keep 2; losses / weights match full checkpointing.
    usable = [8 * GIB, 2 * GIB + GIB // 4]
    auto = _launch("ddp", (static_graph, find_unused, accumulation, True), usable)
    reference = _launch("ddp", (static_graph, find_unused, accumulation, False), usable)
    assert auto[0]["kept"] == auto[1]["kept"]
    assert 0 < len(auto[1]["kept"]) < 4
    assert auto[0]["checkpointed"] == auto[1]["checkpointed"]
    for rank in (0, 1):
        assert reference[rank]["checkpointed"] == [True] * 4
        _assert_same_training(auto[rank], reference[rank])


def _fsdp2_runs_on_cpu():
    # torch 2.6's FSDP2 all-gather records events on a CPU stream that has no record_event.
    return hasattr(torch.cpu.current_stream(), "record_event")


@pytest.mark.skipif(not _fsdp2_runs_on_cpu(), reason = "this torch has no CPU FSDP2 path")
@pytest.mark.parametrize("accumulation", [1, 2])
def test_fsdp2_training_unchanged_and_ranks_agree(accumulation):
    # The collective runs in the root forward pre-hook, next to FSDP2's own all-gathers.
    usable = [8 * GIB, 2 * GIB + GIB // 4]
    auto = _launch("ddp", ("fsdp2", False, accumulation, True), usable)
    reference = _launch("ddp", ("fsdp2", False, accumulation, False), usable)
    assert auto[0]["kept"] == auto[1]["kept"]
    assert 0 < len(auto[1]["kept"]) < 4
    for rank in (0, 1):
        assert reference[rank]["checkpointed"] == [True] * 4
        _assert_same_training(auto[rank], reference[rank])
