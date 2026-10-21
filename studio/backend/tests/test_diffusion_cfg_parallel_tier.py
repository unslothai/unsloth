# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CFG-parallel on the default speed tier and on Z-Image, CPU only (the replica shares the CPU device).

The deferred speed profile arms a whole-step CUDA graph on the denoiser's forward slot, where the CFG-parallel router
sits: the graph must install under the router so the guidance branches keep routing. Z-Image hands its split guided
call to the router as an explicit pair, since its halves are fresh tensors that identity pairing never matches."""

from __future__ import annotations

import threading
import types

import pytest
import torch

from core.inference import diffusion_cuda_graph as cg
from core.inference import diffusion_multigpu as mg


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(mg.MULTIGPU_ENV, raising = False)
    monkeypatch.delenv(mg.CFG_PARALLEL_ENV, raising = False)
    yield


class _Dit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        return_dict = False,
    ):
        return (self.proj(hidden_states) * timestep + encoder_hidden_states.sum(),)


def _router(dit):
    replicas = []

    def factory():
        r = mg.build_replica(dit, torch.device("cpu"))
        replicas.append(r)
        return r

    router = mg.CfgParallelRouter(dit, primary = "cpu", secondary = "cpu", replica_factory = factory)
    dit.forward = router
    dit._unsloth_cfg_router = router
    return router, replicas


def _cfg_loop(dit, latents, steps):
    cond = torch.randn(3, 4, generator = torch.Generator().manual_seed(1))
    uncond = torch.randn(3, 4, generator = torch.Generator().manual_seed(2))
    with torch.no_grad():
        for i in range(steps):
            t = torch.tensor(1.0 - i / steps)
            a = dit(hidden_states = latents, timestep = t, encoder_hidden_states = cond)[0]
            b = dit(hidden_states = latents, timestep = t, encoder_hidden_states = uncond)[0]
            latents = latents - 0.1 * (b + 4.0 * (a - b))
    return latents


class _Pipe:
    def __init__(self, transformer):
        self.transformer = transformer


def test_graph_wrapper_installs_under_the_router_and_branches_keep_routing():
    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    reference = _cfg_loop(dit, latents.clone(), steps = 5)
    router, replicas = _router(dit)
    pipe = _Pipe(dit)
    mg._CFG_ROUTERS[pipe] = router
    # what the deferred speed profile arms on the default tier (not enabled here: a CPU tensor is never captured)
    graph = cg.GraphedForward(dit).install()
    assert dit.__dict__["forward"] is router and router.inner is graph
    assert not mg.drop_displaced_cfg_router(pipe, "test")
    out = _cfg_loop(dit, latents.clone(), steps = 5)
    assert torch.equal(out, reference)
    assert router.stats["steps_parallel"] == 4 and router.stats["replica_calls"] == 4
    assert len(replicas) == 1 and router.disabled_reason is None
    # the primary branches go through the graph wrapper, the replica runs the class forward
    assert "forward" not in replicas[0].__dict__
    graph.uninstall()
    assert router.inner is None and dit.__dict__["forward"] is router
    graph.install()
    mg.uninstall_cfg_parallel(pipe)
    assert dit.__dict__["forward"] is graph


def test_primary_branch_replays_through_the_inner_graph(monkeypatch):
    torch.manual_seed(0)
    dit = _Dit()
    router, _replicas = _router(dit)
    graph = cg.GraphedForward(dit).install()
    seen = []
    eager = graph._eager

    def counting(args, kwargs):
        seen.append(kwargs["encoder_hidden_states"])
        return eager(args, kwargs)

    monkeypatch.setattr(graph, "_eager", counting)
    _cfg_loop(dit, torch.randn(2, 4), steps = 4)
    # step 0 runs both branches on the source (the replica is built at its odd branch), then only the even ones
    assert len(seen) == 4 + 1
    assert router.stats["steps_parallel"] == 3


def test_replica_jobs_and_captures_never_overlap():
    entered = threading.Event()
    release = threading.Event()
    captured = threading.Event()

    def replica_job():
        with cg.outside_capture():
            entered.set()
            release.wait(5)

    def capture():
        with cg._capturing():
            captured.set()

    a = threading.Thread(target = replica_job)
    a.start()
    assert entered.wait(5)
    b = threading.Thread(target = capture)
    b.start()
    assert not captured.wait(0.3)  # the capture waits for the replica forward
    release.set()
    assert captured.wait(5)
    a.join(5)
    b.join(5)

    # and the other way round: a replica job waits for a recording to end
    recording = threading.Event()
    stop = threading.Event()
    ran = threading.Event()

    def long_capture():
        with cg._capturing():
            recording.set()
            stop.wait(5)

    def job():
        with cg.outside_capture():
            ran.set()

    c = threading.Thread(target = long_capture)
    c.start()
    assert recording.wait(5)
    d = threading.Thread(target = job)
    d.start()
    assert not ran.wait(0.3)
    stop.set()
    assert ran.wait(5)
    c.join(5)
    d.join(5)
    assert cg._OUTSIDE_JOBS == 0 and cg._CAPTURE_DEPTH == 0


def test_a_replica_thread_that_records_does_not_wait_on_itself():
    done = threading.Event()

    def job():
        with cg.outside_capture():
            with cg._capturing():
                done.set()

    t = threading.Thread(target = job)
    t.start()
    assert done.wait(5)
    t.join(5)


# Z-Image: the split guided call reaches the router as one explicit pair.

zmod = None


def _zimage():
    global zmod
    if zmod is None:
        import importlib
        try:
            zmod = importlib.import_module("diffusers.models.transformers.transformer_z_image")
        except (ImportError, RuntimeError) as exc:
            pytest.skip(f"diffusers is not importable here: {type(exc).__name__}")
    torch.manual_seed(0)
    return zmod.ZImageTransformer2DModel(
        all_patch_size = (2,),
        all_f_patch_size = (1,),
        in_channels = 16,
        dim = 32,
        n_layers = 2,
        n_refiner_layers = 1,
        n_heads = 2,
        n_kv_heads = 2,
        norm_eps = 1e-5,
        qk_norm = True,
        cap_feat_dim = 16,
        rope_theta = 256.0,
        t_scale = 1000.0,
        axes_dims = [8, 4, 4],
        axes_lens = [512, 64, 64],
    ).eval()


def _zimage_steps(model, pipe, steps):
    from core.inference import diffusion_zimage_cfg_split as zs

    outs = []
    g = torch.Generator().manual_seed(7)
    latent = torch.randn(16, 1, 8, 8, generator = g)
    cond = torch.randn(9, 16, generator = g)
    neg = torch.randn(40, 16, generator = g)
    with torch.no_grad(), zs.render(pipe, 1):
        for i in range(steps):
            # what ZImagePipeline passes on a guided step: fresh repeated latents and timesteps every call
            t = torch.tensor([1.0 - i / steps]).repeat(2)
            out = model([latent.clone(), latent.clone()], t, [cond, neg], return_dict = False)[0]
            latent = latent - 0.1 * (out[1] + 3.0 * (out[0] - out[1]))
            outs.append(latent)
    return outs


@pytest.fixture
def _zs():
    from core.inference import diffusion_zimage_cfg_split as zs
    yield zs
    zs.uninstall_zimage_cfg_split()


def test_zimage_split_runs_its_halves_as_one_parallel_pair(_zs):
    model = _zimage()
    pipe = types.SimpleNamespace(transformer = model)
    assert _zs.install_zimage_cfg_split(pipe, offload_policy = "none") is None
    reference = _zimage_steps(model, pipe, 4)
    router, replicas = _router(model)
    out = _zimage_steps(model, pipe, 4)
    for a, b in zip(out, reference):
        assert torch.equal(a, b)
    assert len(replicas) == 1 and router.disabled_reason is None
    assert router.stats["pairs_parallel"] == 4
    assert router.stats["replica_calls"] == 4 and router.stats["steps_parallel"] == 4
    assert router.stats["primary_calls"] == 4
    status = _zs.zimage_cfg_split_status(pipe)
    assert status["split"] == 8 and status["batched"] == 0 and status["parallel"] == 4


def test_zimage_pair_falls_back_to_the_source_when_the_replica_fails(_zs):
    model = _zimage()
    pipe = types.SimpleNamespace(transformer = model)
    _zs.install_zimage_cfg_split(pipe, offload_policy = "none")
    reference = _zimage_steps(model, pipe, 3)
    router, replicas = _router(model)

    def broken():
        raise RuntimeError("replica out of memory")

    router.replica_factory = broken
    out = _zimage_steps(model, pipe, 3)
    for a, b in zip(out, reference):
        assert torch.equal(a, b)
    assert router.disabled_reason is not None and router.stats["pairs_parallel"] == 0


def test_a_graph_armed_at_load_moves_under_a_later_router(monkeypatch):
    # speed tier armed at load (no deferral): the graph is in the slot first, then install_cfg_parallel runs
    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    reference = _cfg_loop(dit, latents.clone(), steps = 4)
    graph = cg.GraphedForward(dit).install()
    assert mg._hooks_present(dit) is None
    probes = {
        o: mg.DeviceProbe(ordinal = o, name = "T4", total_mib = 15000, free_mib = 14000) for o in (0, 1)
    }
    monkeypatch.setattr(mg, "cfg_parallel_decline", lambda pipe, **kw: None)
    monkeypatch.setattr(mg, "_module_device", lambda m: torch.device("cuda", 0))
    monkeypatch.setattr(
        mg, "probe_devices", lambda ordinals, **kw: {o: probes[o] for o in ordinals}
    )
    pipe = _Pipe(dit)
    try:
        with mg.selection_scope(mg.MultiGpuSelection(primary = 0, secondaries = (1,))):
            assert mg.install_cfg_parallel(pipe, offload_policy = "none") is None
        router = mg._CFG_ROUTERS[pipe]
        assert (
            dit.__dict__["forward"] is router and router.inner is graph and not router.had_override
        )
        # run the routing on the CPU (the probe above only placed it)
        router.primary = router.secondary = torch.device("cpu")
        router.replica_factory = lambda: mg.build_replica(dit, torch.device("cpu"))
        out = _cfg_loop(dit, latents.clone(), steps = 4)
        assert torch.equal(out, reference)
        assert router.stats["steps_parallel"] == 3 and router.disabled_reason is None
    finally:
        mg.uninstall_cfg_parallel(pipe)
    assert dit.__dict__["forward"] is graph


def test_a_queued_replica_job_holds_captures_off_until_it_has_dropped_its_inputs():
    worker = mg._ReplicaWorker(torch.device("cpu"))
    worker.call(lambda: None)  # thread started and bound
    assert cg._OUTSIDE_JOBS == 0
    gate = threading.Event()
    dropped = []

    class Input:
        def __del__(self):
            dropped.append(cg._OUTSIDE_JOBS)

    def job(x = Input()):
        gate.wait(5)
        return 1

    # held from submit, before the worker even picks it up
    done = worker.submit(job)
    del job
    assert cg._OUTSIDE_JOBS == 1
    captured = threading.Event()

    def capture():
        with cg._capturing():
            captured.set()

    t = threading.Thread(target = capture)
    t.start()
    assert not captured.wait(0.3)
    gate.set()
    assert mg._ReplicaWorker.wait(done) == 1
    assert captured.wait(5)
    t.join(5)
    # the job's inputs were released while the hold was still counted
    assert dropped == [1]
    assert cg._OUTSIDE_JOBS == 0
    worker.close()


def test_a_graph_installs_under_a_router_nested_in_a_static_step_skip():
    # deferred speed: the auto static skip wraps the router armed at load, then the graph arms
    from core.inference.diffusion_step_skip import StaticStepSkip

    torch.manual_seed(0)
    dit = _Dit()
    latents = torch.randn(2, 4)
    reference = _cfg_loop(dit, latents.clone(), steps = 4)
    router, _replicas = _router(dit)
    pipe = _Pipe(dit)
    mg._CFG_ROUTERS[pipe] = router
    skip = StaticStepSkip(dit, inner = router)
    dit.__dict__["forward"] = skip
    graph = cg.GraphedForward(dit).install()
    # the graph goes beneath the router, not over it
    assert dit.__dict__["forward"] is skip and skip.inner is router and router.inner is graph
    graph.uninstall()
    assert router.inner is None and skip.inner is router
    graph.install()
    # a replica cannot carry the skip: the router is dropped and the skip keeps the graph
    assert mg.drop_displaced_cfg_router(pipe, "test")
    assert mg._CFG_DECLINES[pipe] == "a static step skip wraps the denoiser"
    assert pipe not in mg._CFG_ROUTERS
    assert dit.__dict__["forward"] is skip and skip.inner is graph and router.inner is None
    assert torch.equal(_cfg_loop(dit, latents.clone(), steps = 4), reference)


def test_a_worker_that_fails_to_start_releases_its_capture_hold(monkeypatch):
    assert cg._OUTSIDE_JOBS == 0

    def refuse(self):
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", refuse)
    with pytest.raises(RuntimeError):
        mg._ReplicaWorker(torch.device("cpu"))
    assert cg._OUTSIDE_JOBS == 0


def test_captures_stay_held_off_while_a_replica_result_is_copied_home(_zs, monkeypatch):
    model = _zimage()
    pipe = types.SimpleNamespace(transformer = model)
    _zs.install_zimage_cfg_split(pipe, offload_policy = "none")
    router, _replicas = _router(model)
    held = []
    move = mg.move_to_device

    def recording(obj, device, *a, **kw):
        if device == router.primary:
            held.append(cg._OUTSIDE_JOBS)
        return move(obj, device, *a, **kw)

    monkeypatch.setattr(mg, "move_to_device", recording)
    _zimage_steps(model, pipe, 3)
    assert router.stats["pairs_parallel"] == 3
    assert held and all(n >= 1 for n in held)
    assert cg._OUTSIDE_JOBS == 0
