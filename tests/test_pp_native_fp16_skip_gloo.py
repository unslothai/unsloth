# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""float16 GradScaler skip agreement over pipeline stages when the caller uses the plain torch AMP recipe.
An inf on one stage must skip the step and back off the scale on every stage."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pickle  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pp_native_gloo import LLAMA, _free_port, _torch_pipelining_error, batch, build, ppn  # noqa: E402

RECIPES = ("pp_unscale", "scaler_unscale", "scaler_step_only")


def _one_step(rank, world, recipe, inf_rank):
    from torch.distributed.device_mesh import init_device_mesh

    model = build(LLAMA)
    ids, labels = batch(model)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))
    pp = ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = 2)
    scaler = torch.amp.GradScaler("cpu", init_scale = 1024.0)
    opt = torch.optim.SGD(pp.parameters(), lr = 1.0)
    pp.step(ids, labels, scaler = scaler)
    if rank == inf_rank:
        pp.parameters()[0].grad.view(-1)[0] = float("inf")
    before = [p.detach().clone() for p in pp.parameters()]
    if recipe == "pp_unscale":
        pp.unscale_(scaler, opt)
        pp.clip_grad_norm_(1.0)
    elif recipe == "scaler_unscale":
        scaler.unscale_(opt)
        pp.clip_grad_norm_(1.0)
    scaler.step(opt)
    scaler.update()
    after = [p.detach().clone() for p in pp.parameters()]
    try:
        pickle.loads(pickle.dumps(scaler))  # torch.amp allows pickling a GradScaler after update()
        picklable = "ok"
    except Exception as exc:
        picklable = f"{type(exc).__name__}: {exc}"
    return {
        "unchanged": all(torch.equal(a, b) for a, b in zip(before, after)),
        "scale": float(scaler.get_scale()),
        "picklable": picklable,
    }


def _clip_before_unscale(rank, world):
    from torch.distributed.device_mesh import init_device_mesh

    model = build(LLAMA)
    ids, labels = batch(model)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))
    pp = ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = 2)
    scaler = torch.amp.GradScaler("cpu", init_scale = 1024.0)
    opt = torch.optim.SGD(pp.parameters(), lr = 1.0)
    out = {}
    pp.step(ids, labels, scaler = scaler)
    try:
        pp.clip_grad_norm_(1.0)
        out["scaled_clip"] = None
    except ValueError as exc:
        out["scaled_clip"] = str(exc)
    pp.unscale_(scaler, opt)
    out["unscaled_clip"] = float(pp.clip_grad_norm_(1.0))
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none = True)
    # A step without a scaler clips as before.
    pp.step(ids, labels)
    out["no_scaler_clip"] = float(pp.clip_grad_norm_(1.0))
    return out


def _clip_after_step(rank, world):
    """Gradients are unscaled once scaler.step ran (it unscales first) and stay so after update():
    a grad-norm read there (logging) is correct usage and must not be refused."""
    from torch.distributed.device_mesh import init_device_mesh

    model = build(LLAMA)
    ids, labels = batch(model)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("pp",))
    pp = ppn.apply_pipeline_parallel(model, mesh["pp"], n_microbatches = 2)
    scaler = torch.amp.GradScaler("cpu", init_scale = 1024.0)
    opt = torch.optim.SGD(pp.parameters(), lr = 1.0)
    out = {}
    pp.step(ids, labels, scaler = scaler)
    for where, action in (
        ("after_step", lambda: scaler.step(opt)),
        ("after_update", scaler.update),
    ):
        action()
        try:
            out[where] = float(pp.clip_grad_norm_(1e9))
        except ValueError as exc:
            out[where] = str(exc)
    return out


def _worker(rank, world, port, tmp, case, kwargs):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 180)
    )
    try:
        fn = {"step": _one_step, "clip": _clip_before_unscale, "clip_after": _clip_after_step}[case]
        res = fn(rank, world, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _spawn(
    case,
    world = 2,
    **kwargs,
):
    if _torch_pipelining_error():
        pytest.skip(_torch_pipelining_error())
    ctx = multiprocessing.get_context("spawn")
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, tmp, case, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


@pytest.mark.parametrize("recipe", RECIPES)
@pytest.mark.parametrize("inf_rank", [0, 1])
def test_pp2_inf_on_one_stage_skips_every_stage(recipe, inf_rank):
    results = _spawn("step", recipe = recipe, inf_rank = inf_rank)
    assert [r["unchanged"] for r in results] == [True, True], (recipe, inf_rank, results)
    assert [r["scale"] for r in results] == [512.0, 512.0], (recipe, inf_rank, results)


@pytest.mark.parametrize("recipe", RECIPES)
def test_pp2_finite_step_updates_every_stage(recipe):
    results = _spawn("step", recipe = recipe, inf_rank = -1)
    assert [r["unchanged"] for r in results] == [False, False], (recipe, results)
    assert [r["scale"] for r in results] == [1024.0, 1024.0], (recipe, results)


def test_pp2_clip_before_unscale_is_refused():
    for r in _spawn("clip"):
        assert r["scaled_clip"] is not None and "unscale_" in r["scaled_clip"], r
        assert r["unscaled_clip"] > 0 and r["no_scaler_clip"] > 0, r


def test_pp2_scaler_still_pickles_after_update():
    """pp.step(scaler = ...) must not leave the scaler unpicklable (torch.save(scaler) after scaler.update())."""
    results = _spawn("step", recipe = "scaler_step_only", inf_rank = -1)
    assert [r["picklable"] for r in results] == ["ok", "ok"], results


def test_pp2_clip_after_scaler_step_is_allowed():
    for r in _spawn("clip_after"):
        assert all(isinstance(v, float) and v > 0 for v in r.values()), r
