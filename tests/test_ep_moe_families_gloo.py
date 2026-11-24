# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Expert parallel (gloo CPU, 2 ranks, fp32) on tiny MoE families vs one process on the concatenated batch.

Each rank trains its own rows (rank 1 a shorter sequence: uneven all-to-all splits) through torch DDP with the
routed experts sharded by index; LoRA on every Linear (the routers included) and on the fused expert stacks.
Families whose experts the tensor parallel expert shard cannot cut (gpt-oss: another experts call signature) or
models without routed experts are refused before anything changes."""

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

FAMILIES = {
    "qwen3_moe": "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
    "olmoe": "hf-internal-testing/tiny-random-OlmoeForCausalLM",
    "glm4_moe": "trl-internal-testing/tiny-Glm4MoeForCausalLM",  # shared experts + grouped routing
}
REFUSED = {
    "gpt_oss": "trl-internal-testing/tiny-GptOssForCausalLM",
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
}
EXPERT_LORA = dict(target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"])
_SEQ = (7, 5, 6)


def _modules():
    from test_tp_native_oracle import build, tpn

    from test_tp_inkling2_gloo import _load
    return build, tpn, _load("ep_native")


def _rows(vocab, rank):
    ids = torch.randint(
        0, vocab, (2, _SEQ[rank]), generator = torch.Generator().manual_seed(20 + rank)
    )
    labels = ids.clone()
    labels[0, :2] = -100
    return ids, labels


def _sum_loss(model, ids, labels):
    logits = model(input_ids = ids, use_cache = False).logits
    return torch.nn.functional.cross_entropy(
        logits[:, :-1].flatten(0, 1).float(),
        labels[:, 1:].flatten(),
        ignore_index = -100,
        reduction = "sum",
    )


def _build(build, repo):
    model = build(repo, **EXPERT_LORA).train()
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():  # lora_B = 0 would leave every lora_A gradient of the first step at 0
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator = g) * 0.02)
    return model


def _case_family(rank, world, tmp, repo):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel

    build, tpn, epn = _modules()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ref = _build(build, repo)
    vocab = tpn._text_config(ref).vocab_size
    rows = [_rows(vocab, r) for r in range(world)]
    n = sum(int((lab[:, 1:] != -100).sum()) for _, lab in rows)
    # One process: one backward per rank's rows, the token mean over all of them.
    for ids, lab in rows:
        (_sum_loss(ref, ids, lab) / n).backward()
    want = {k: p.grad.detach().clone() for k, p in ref.named_parameters() if p.grad is not None}
    want_loss = sum(float(_sum_loss(ref, i, lab).detach()) for i, lab in rows) / n
    want_norm = float(
        torch.nn.utils.clip_grad_norm_([p for p in ref.parameters() if p.grad is not None], 1e9)
    )

    model = _build(build, repo)
    epn.apply_expert_parallel(model, mesh)
    group = mesh.get_group()
    ddp = DistributedDataParallel(model)
    loss = _sum_loss(ddp, *rows[rank]) * world / n
    loss.backward()
    value = loss.detach() / world
    dist.all_reduce(value)
    got = {
        k: (epn._gather_expert(p, group, p.grad) if getattr(p, epn._EP, None) else p.grad)
        .detach()
        .clone()
        for k, p in model.named_parameters()
        if p.grad is not None
    }
    norm = float(
        epn.ep_clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], 1e9, group = group
        )
    )
    shard = tpn.moe_blocks_with_experts(model)[0].__dict__["_unsloth_tp_expert_shard"]
    return {
        "loss": (float(value), want_loss),
        "norm": (norm, want_norm),
        "grads": (got, want),
        "per_rank": shard.per_rank,
        "experts": shard.experts,
        "blocks": len(tpn.moe_blocks_with_experts(model)),
    }


def _case_refused(rank, world, tmp, repo):
    from torch.distributed.device_mesh import init_device_mesh

    build, tpn, epn = _modules()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    model = build(repo, lora = False)
    before = {k: (id(p), tuple(p.shape)) for k, p in model.named_parameters()}
    try:
        epn.apply_expert_parallel(model, mesh)
        message = None
    except ValueError as exc:
        message = str(exc)
    after = {k: (id(p), tuple(p.shape)) for k, p in model.named_parameters()}
    return {
        "message": message,
        "unchanged": before == after,
        "is_ep": epn.is_expert_parallel(model),
    }


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
    except Exception as exc:
        import traceback
        res = {"error": f"{type(exc).__name__}: {exc}"[:600] + traceback.format_exc()[-4000:]}
    torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
    dist.destroy_process_group()


def _spawn(
    case,
    world = 2,
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


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_ep_family_matches_one_process(family):
    _check_family(_spawn("_case_family", repo = FAMILIES[family]), 2)


def test_ep_three_ranks_pad_experts_and_leave_one_rank_only_padding():
    # 4 experts over 3 ranks: 2 per rank, padded to 6, so rank 2 owns only zero experts the router never picks
    # (every call it receives no token: the empty-receiver path) and the splits are uneven three ways.
    _check_family(_spawn("_case_family", world = 3, repo = FAMILIES["olmoe"]), 3)


def _check_family(results, world):
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["blocks"] >= 1 and r["per_rank"] == max(2, -(-r["experts"] // world)), r
        got, want = r["loss"]
        assert got == pytest.approx(want, rel = 1e-6), r["loss"]
        got, want = r["norm"]
        assert got == pytest.approx(want, rel = 1e-5), r["norm"]
        got, want = r["grads"]
        assert set(got) == set(want), sorted(set(got) ^ set(want))[:6]
        assert any(".experts." in k and "lora_" in k for k in want)
        for k in want:
            torch.testing.assert_close(got[k], want[k], rtol = 1e-5, atol = 1e-6, msg = k)


@pytest.mark.parametrize("family", sorted(REFUSED))
def test_ep_refuses_models_it_cannot_shard(family):
    results = _spawn("_case_refused", repo = REFUSED[family])
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["message"] is not None and "no shardable MoE block" in r["message"], r["message"]
        assert r["unchanged"] and not r["is_ep"], r
