# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""ep_sync_expert_grads in bounded buckets (gloo CPU, 4 ranks = ep 2 x 2 replicas): bitwise the one-buffer sum."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))

pytest.importorskip("transformers.core_model_loading")

REPO = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
CAP = 4096  # bytes: many buckets, and expert stacks larger than one bucket


def _modules():
    from test_tp_native_oracle import build, tpn
    from unsloth_models_loader import load_models_module
    return build, tpn, load_models_module("ep_native")


def _one_buffer_sum(epn, model):
    """The previous ep_sync_expert_grads: one flat buffer per (device, dtype)."""
    import torch.distributed as dist
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    grads = []
    for _, p in epn._ep_params(model):
        if p.requires_grad:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            grads.append(p.grad)
    buckets = {}
    for g in grads:
        buckets.setdefault((g.device, g.dtype), []).append(g)
    with torch.no_grad():
        for tensors in buckets.values():
            flat = _flatten_dense_tensors(tensors)
            dist.all_reduce(flat, group = epn._replicas(model).get_group())
            for t, v in zip(tensors, _unflatten_dense_tensors(flat, tensors)):
                t.copy_(v)


def _model(build, epn, mesh):
    model = build(
        REPO, target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    ).train()
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator = g) * 0.02)
    epn.apply_expert_parallel(model, mesh)
    return model


def _case_buckets(rank, world, tmp):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel

    build, tpn, epn = _modules()
    epn._REPLICA_BUCKET_BYTES = CAP
    mesh = init_device_mesh("cpu", (world // 2, 2), mesh_dim_names = ("ep_rep", "ep"))
    models = [_model(build, epn, mesh) for _ in range(2)]
    vocab = tpn._text_config(models[0]).vocab_size
    out = {"steps": [], "max_staging": 0, "in_place": 0}
    grads_of = set()
    real = dist.all_reduce

    def probe(tensor, *args, **kwargs):
        if id(tensor) in grads_of:
            out["in_place"] += 1
        else:
            out["max_staging"] = max(out["max_staging"], tensor.numel() * tensor.element_size())
        return real(tensor, *args, **kwargs)

    for step in range(2):
        ids = torch.randint(
            0, vocab, (2, 6), generator = torch.Generator().manual_seed(100 * step + rank)
        )
        result = []
        for i, model in enumerate(models):
            ddp = DistributedDataParallel(model)
            ddp(input_ids = ids, labels = ids, use_cache = False).loss.backward()
            if i == 0:
                grads_of = {id(p.grad) for _, p in epn._ep_params(model) if p.grad is not None}
                dist.all_reduce = probe
                try:
                    epn.ep_sync_expert_grads(model)
                finally:
                    dist.all_reduce = real
            else:
                _one_buffer_sum(epn, model)
            grads = {k: p.grad.clone() for k, p in epn._ep_params(model) if p.grad is not None}
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.sub_(0.1 * p.grad)
                        p.grad = None
            after = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}
            result.append((grads, after))
            del ddp
        (ga, pa), (gb, pb) = result
        out["steps"].append(
            {
                "expert_grads": len(ga),
                "grads_equal": set(ga) == set(gb) and all(torch.equal(ga[k], gb[k]) for k in ga),
                "params_equal": set(pa) == set(pb) and all(torch.equal(pa[k], pb[k]) for k in pa),
                "largest_grad": max(g.numel() * g.element_size() for g in ga.values()),
            }
        )

    # A gradient left None (an expert stack no token reached on any step) is summed as zeros.
    trained = [q for _, q in epn._ep_params(models[0]) if q.requires_grad]
    p = trained[0]
    for q in trained:
        q.grad = None if q is p else torch.ones_like(q)
    epn.ep_sync_expert_grads(models[0])
    out["none_grad_zero"] = p.grad is not None and not p.grad.any()
    out["others_summed"] = all(torch.equal(q.grad, torch.full_like(q, 2.0)) for q in trained[1:])
    return out


def _worker(rank, world, tmp, case, kwargs):
    os.environ.update(
        RANK = str(rank), WORLD_SIZE = str(world), LOCAL_RANK = str(rank), LOCAL_WORLD_SIZE = str(world)
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(1)
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
    world = 4,
    **kwargs,
):
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


def test_replica_sum_in_bounded_buckets_is_bitwise_the_one_buffer_sum():
    results = _spawn("_case_buckets")
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        for step in r["steps"]:
            assert step["expert_grads"] > 0 and step["grads_equal"] and step["params_equal"], step
            assert step["largest_grad"] > CAP, step  # an oversized stack is reduced in place
        assert r["in_place"] > 0, r
        assert 0 < r["max_staging"] <= CAP, r["max_staging"]
        assert r["none_grad_zero"] and r["others_summed"], r
