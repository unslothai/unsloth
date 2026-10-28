# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""dp 2 x tp 2 gloo: a TP param in FSDP2 ignored_params is averaged over dp; replicas match a single process."""

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

_MAX_NORM = 0.05  # small enough that every step clips


def _ignored_names(model):
    return [
        n
        for n, p in model.named_parameters()
        if p.requires_grad and "layers.0." in n and ("q_proj" in n or "down_proj" in n)
    ]


def _case_tp_fsdp2_ignored(rank, world, tmp):
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
    names = _ignored_names(model)
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


def _spawn(world):
    """test_tp_native_gloo._spawn with this module's case registered in every spawned rank."""
    import tempfile

    ctx = multiprocessing.get_context("spawn")
    port = gloo._free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, "tp_fsdp2_ignored", tmp, {}))
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


def test_tp_param_in_ignored_params_is_averaged_over_dp():
    results = _spawn(4)
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
