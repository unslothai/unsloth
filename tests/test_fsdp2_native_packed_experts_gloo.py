# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 over zoo packed expert stacks (NVFP4, MXFP8, MXFP4), gloo world of 2: the stacks stay
replicated (fully_shard cannot take uint8 subclasses), the rest is sharded, LoRA grads match one process."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
import test_ep_packed_experts_gloo as ep  # noqa: E402


def _clean(name):
    return name.replace("_checkpoint_wrapped_module.", "")


def _case_fsdp2_packed(
    rank,
    world,
    tmp,
    fmt,
    shard_frozen,
    refuse = False,
):
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from unsloth_models_loader import load_models_module

    ep._zoo_backend()
    native = load_models_module("fsdp2_native")
    import unsloth_zoo.packed_experts as pe

    ref = ep._lora(_quantized(fmt))
    rows = [ep._rows(128, r) for r in range(world)]
    n = sum(int((lab[:, 1:] != -100).sum()) for _, lab in rows)
    for ids, lab in rows:
        (ep._sum_loss(ref, ids, lab) / n).backward()
    want = {
        _clean(k): p.grad.detach().clone() for k, p in ref.named_parameters() if p.grad is not None
    }
    with torch.no_grad():
        want_loss = sum(float(ep._sum_loss(ref, i, lab)) for i, lab in rows) / n

    model = ep._lora(_quantized(fmt))
    if refuse:
        native._supports_ignored_params = lambda fully_shard: False
    try:
        model = native.shard_model(
            model, native.FSDP2Config(param_dtype = torch.float32, shard_frozen = shard_frozen)
        )
    except Exception as exc:  # noqa: BLE001
        return {"shard": f"{type(exc).__name__}: {str(exc)[:400]}"}
    stacks = [p for n_, p in model.named_parameters() if ".experts." in n_ and "lora_" not in n_]
    calls = ep._uses()
    loss = ep._sum_loss(model, *rows[rank]) * world / n
    loss.backward()
    engaged = ep._uses() - calls
    value = loss.detach() / world
    dist.all_reduce(value)
    got = {}
    for k, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        if isinstance(g, DTensor):
            g = g.full_tensor()
        else:
            g = g.clone()
            dist.all_reduce(g)
        got[_clean(k)] = g.detach().clone()
    return {
        "shard": "ok",
        "stacks": len(stacks),
        "packed": sum(pe.is_packed_expert_param(p) and not isinstance(p, DTensor) for p in stacks),
        "sharded": sum(isinstance(p, DTensor) for p in model.parameters()),
        "engaged": engaged,
        "loss": (float(value), want_loss),
        "grads": (got, want),
    }


def _quantized(fmt):
    model = ep._base()
    ep._quantize(model, fmt)
    return model


def _worker(rank, world, tmp, case, kwargs):
    os.environ.update(
        RANK = str(rank), WORLD_SIZE = str(world), LOCAL_RANK = str(rank), LOCAL_WORLD_SIZE = str(world)
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(2)
    dist.init_process_group(
        "gloo",
        store = dist.FileStore(os.path.join(tmp, "rendezvous"), world),
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = 300),
    )
    try:
        res = globals()[case](rank, world, tmp, **kwargs)
    except BaseException as exc:  # a rank must report, not vanish
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}"[:600] + traceback.format_exc()[-4000:]}
    torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    dist.destroy_process_group()


def _spawn(
    case,
    world = 2,
    **kwargs,
):
    _requirements()
    ctx = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, tmp, case, kwargs)) for r in range(world)
        ]
        for p in procs:
            p.start()
        deadline = time.monotonic() + 600
        while any(p.is_alive() for p in procs) and time.monotonic() < deadline:
            time.sleep(0.2)
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(30)
        out = []
        for r in range(world):
            path = os.path.join(tmp, f"result_{r}.pt")
            out.append(torch.load(path, weights_only = False) if os.path.exists(path) else None)
        return out


def _requirements():
    ep._requirements()
    try:
        from torch.distributed.fsdp import fully_shard  # noqa: F401
    except ImportError:
        pytest.skip("torch without FSDP2 fully_shard")


@pytest.mark.parametrize(
    "fmt, shard_frozen",
    [("nvfp4", True), ("mxfp8", True), ("mxfp4", True), ("nvfp4", None)],
)
def test_fsdp2_keeps_packed_expert_stacks_replicated(fmt, shard_frozen):
    import inspect

    from torch.distributed.fsdp import fully_shard

    if "ignored_params" not in inspect.signature(fully_shard).parameters:
        pytest.skip("torch without fully_shard(ignored_params) (2.7+): refused, see the test below")
    results = _spawn("_case_fsdp2_packed", fmt = fmt, shard_frozen = shard_frozen)
    for rank, r in enumerate(results):
        assert r is not None and "error" not in r, r
        assert r["shard"] == "ok", (rank, r["shard"])
        assert r["stacks"] == 4 and r["packed"] == 4, (rank, r["stacks"], r["packed"])
        assert r["sharded"] > 0, rank
        assert r["engaged"] > 0, (rank, r["engaged"])
        got, want = r["loss"]
        assert got == pytest.approx(want, rel = 1e-5), r["loss"]
        got, want = r["grads"]
        assert set(got) == set(want), sorted(set(got) ^ set(want))[:6]
        assert any(".experts." in k and "lora_" in k for k in want)
        for k in want:
            torch.testing.assert_close(got[k], want[k], rtol = 1e-4, atol = 1e-6, msg = k)


def test_fsdp2_packed_expert_stacks_refused_without_ignored_params():
    for r in _spawn("_case_fsdp2_packed", fmt = "nvfp4", shard_frozen = True, refuse = True):
        assert r is not None and "error" not in r, r
        assert r["shard"].startswith("RuntimeError") and "ignored_params" in r["shard"], r["shard"]
        assert "packed" in r["shard"], r["shard"]
