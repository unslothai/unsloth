# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ignored TP trainables stay synchronized over DP, including empty FSDP2 groups."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_oracle import LLAMA, _batch, build, tpn  # noqa: E402
import test_tp_native_gloo as gloo  # noqa: E402

_MAX_NORM = 0.05


def _ignored_names(model, ignore_all = False):
    return [
        n
        for n, p in model.named_parameters()
        if p.requires_grad
        and (ignore_all or ("layers.0." in n and ("q_proj" in n or "down_proj" in n)))
    ]


def _case_tp_fsdp2_ignored(
    rank,
    world,
    tmp,
    ignore_all = False,
):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor
    from test_fsdp2_native_units import _load_native

    native = _load_native()
    if not native._supports_ignored_params(native._fsdp_api()[2]):
        return {"unsupported": True}
    mesh = init_device_mesh("cpu", (2, world // 2), mesh_dim_names = ("dp", "tp"))
    dp = mesh["dp"].get_local_rank()
    ref = build(LLAMA)
    model = build(LLAMA)
    ids = _batch(ref, bsz = 4)
    mine = ids[2 * dp : 2 * dp + 2]
    tpn.apply_tensor_parallel(model, mesh["tp"])
    names = _ignored_names(model, ignore_all)
    by_name = dict(model.named_parameters())
    ignored = {by_name[n] for n in names}
    native.shard_model(
        model,
        native.FSDP2Config(param_dtype = torch.float32, mesh = mesh["dp"], ignored_params = ignored),
    )
    by_name = dict(model.named_parameters())
    kept_tp = [
        n for n in names if isinstance(by_name[n], DTensor) and by_name[n].device_mesh.ndim == 1
    ]
    params = [p for p in model.parameters() if p.requires_grad]
    ref_by_name = dict(ref.named_parameters())
    ref_params = [p for p in ref.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr = 1.0)
    ref_opt = torch.optim.SGD(ref_params, lr = 1.0)
    local, norms, ref_norms = [], [], []
    for _ in range(3):
        model(input_ids = mine, labels = mine, use_cache = False).loss.backward()
        ref(input_ids = ids, labels = ids, use_cache = False).loss.backward()
        norms.append(float(native.native_clip_grad_norm_(params, _MAX_NORM)))
        ref_norms.append(float(torch.nn.utils.clip_grad_norm_(ref_params, _MAX_NORM)))
        opt.step(), opt.zero_grad(set_to_none = True)
        ref_opt.step(), ref_opt.zero_grad(set_to_none = True)
        local.append({n: by_name[n].to_local().detach().clone() for n in kept_tp})
    full = {n: tpn._full(by_name[n]).detach().clone() for n in kept_tp}
    dist.barrier()
    return {
        "dp": dp,
        "tp": mesh["tp"].get_local_rank(),
        "names": names,
        "kept_tp": kept_tp,
        "local": local,
        "full": full,
        "ref": {n: ref_by_name[n].detach().clone() for n in kept_tp},
        "norms": norms,
        "ref_norms": ref_norms,
    }


gloo.CASES["tp_fsdp2_ignored"] = _case_tp_fsdp2_ignored


def _case_tp_fsdp2_ignored_compiled(rank, world, tmp):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import ColwiseParallel, parallelize_module
    from torch._dynamo.utils import counters
    from test_fsdp2_native_units import _load_native
    from unsloth.models import fsdp2_save as save

    native = _load_native()
    fully_shard = native._fsdp_api()[2]
    if not native._supports_ignored_params(fully_shard):
        return {"unsupported": True}
    mesh = init_device_mesh("cpu", (2, world // 2), mesh_dim_names = ("dp", "tp"))
    model = torch.nn.Linear(4, 4, bias = False)
    parallelize_module(model, mesh["tp"], ColwiseParallel())
    fully_shard(model, mesh = mesh["dp"], ignored_params = set(model.parameters()))
    model._unsloth_fsdp2_mesh = mesh["dp"]
    names = save.reduce_unsharded_trainable_grads(model)
    compiled = torch.compile(model, backend = "eager", dynamic = True)
    counters.clear()
    for batch in (2, 3, 2):
        x = torch.full((batch, 4), float(mesh["dp"].get_local_rank() + 1))
        compiled(x).sum().backward()
    return {
        "names": names,
        "grad": model.weight.grad.to_local().detach().clone(),
        "graphs": counters["stats"]["unique_graphs"],
        "breaks": list(counters["graph_break"]),
    }


gloo.CASES["tp_fsdp2_ignored_compiled"] = _case_tp_fsdp2_ignored_compiled


def _spawn(
    world,
    case = "tp_fsdp2_ignored",
    **kwargs,
):
    import tempfile

    ctx = multiprocessing.get_context("spawn")
    port = gloo._free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, kwargs))
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


def _worker(*args):
    gloo._worker(*args)


@pytest.mark.parametrize("ignore_all", [False, True], ids = ["some_ignored", "all_ignored"])
def test_tp_param_in_ignored_params_is_averaged_over_dp(ignore_all):
    results = _spawn(4, ignore_all = ignore_all)
    if results[0].get("unsupported"):
        pytest.skip("torch < 2.7: no fully_shard(ignored_params)")
    for res in results:
        assert res["names"] and res["kept_tp"] == res["names"], (res["names"], res["kept_tp"])
    for a in results:
        for b in results:
            if a["tp"] == b["tp"] and a["dp"] < b["dp"]:
                for step, (la, lb) in enumerate(zip(a["local"], b["local"])):
                    for n in la:
                        assert torch.equal(la[n], lb[n]), f"dp replicas differ at step {step}: {n}"
    for res in results:
        assert res["norms"] == pytest.approx(res["ref_norms"], rel = 1e-5), (
            res["norms"],
            res["ref_norms"],
        )
        for n, ref in res["ref"].items():
            torch.testing.assert_close(res["full"][n], ref, rtol = 1e-4, atol = 1e-6, msg = n)


def test_all_ignored_compiled_grad_sync_does_not_add_a_graph_break():
    results = _spawn(4, case = "tp_fsdp2_ignored_compiled")
    if results[0].get("unsupported"):
        pytest.skip("torch < 2.7: no fully_shard(ignored_params)")
    for res in results:
        assert res["names"] == ["weight"]
        assert res["graphs"] > 0
        assert not any("_current_graph_task_id" in reason for reason in res["breaks"]), res[
            "breaks"
        ]
        torch.testing.assert_close(res["grad"], torch.full_like(res["grad"], 1.5 * sum((2, 3, 2))))
