# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Two adjacent pipeline stages on one gloo rank (loop schedules at pp 1): torch < 2.12 warms up with a send to
this rank, which gloo cannot pair ("Pair is not connected"); unsloth's adjacent-stage class drops it."""

import datetime
import os
import sys

import pytest
import torch
import torch.distributed as dist

pipelining = pytest.importorskip("torch.distributed.pipelining")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pp_native_gloo import _free_port, ppn  # noqa: E402  (pp_native from this tree, without importing unsloth)


@pytest.fixture
def gloo_world1(monkeypatch):
    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    # monkeypatch, not os.environ: a MASTER_ADDR / MASTER_PORT left behind lets a later test's env:// rendezvous
    # wait for ranks that never come instead of failing fast.
    if not os.environ.get("MASTER_ADDR"):
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", str(_free_port()))
    dist.init_process_group("gloo", rank = 0, world_size = 1, timeout = datetime.timedelta(seconds = 60))
    try:
        yield dist.group.WORLD
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("schedule", ["ScheduleInterleaved1F1B", "ScheduleLoopedBFS"])
def test_adjacent_stages_on_one_gloo_rank_step(gloo_world1, schedule, train):
    torch.manual_seed(0)
    layers = [torch.nn.Linear(8, 8) for _ in range(2)]
    x = torch.randn(4, 8)
    target = torch.randn(4, 8)
    with torch.no_grad():
        ref = torch.nn.functional.mse_loss(layers[1](layers[0](x)), target)
    cls = ppn._adjacent_stage_class(not train)
    stages = [
        cls(layer, i, 2, torch.device("cpu"), group = gloo_world1) for i, layer in enumerate(layers)
    ]
    # Only torch without same-rank detection needs the warm-up filter, and it must drop every self send there.
    has_same_rank = hasattr(pipelining.PipelineStage, "_is_same_rank")
    assert ("_get_init_p2p_neighbors_ops" in cls.__dict__) is not has_same_rank
    if not has_same_rank:
        assert stages[0]._get_init_p2p_neighbors_ops() == []
    sched = getattr(pipelining, schedule)(
        stages, n_microbatches = 2, loss_fn = torch.nn.functional.mse_loss
    )
    losses = []
    if train:
        sched.step(x, target = target, losses = losses)
    else:
        with torch.no_grad():
            sched.eval(x, target = target, losses = losses) if hasattr(sched, "eval") else sched.step(
                x, target = target, losses = losses
            )
    assert len(losses) == 2
    assert float(torch.stack(losses).mean()) == pytest.approx(float(ref), rel = 1e-5)
