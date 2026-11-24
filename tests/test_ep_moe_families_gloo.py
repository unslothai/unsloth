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
    # (hidden_states, router_indices, routing_weights), transposed stacks, per-expert biases
    "gpt_oss": "trl-internal-testing/tiny-GptOssForCausalLM",
}
REFUSED = {
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
}
EXPERT_LORA = dict(target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"])
_SEQ = (7, 5, 6, 4)


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


def _case_family(
    rank,
    world,
    tmp,
    repo,
    ep = None,
):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel

    build, tpn, epn = _modules()
    ep = ep or world
    if ep == world:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    else:  # replica groups of ep ranks, ep innermost
        mesh = init_device_mesh("cpu", (world // ep, ep), mesh_dim_names = ("ep_rep", "ep"))
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
    group = epn._mesh(model).get_group()
    ddp = DistributedDataParallel(model)
    loss = _sum_loss(ddp, *rows[rank]) * world / n
    loss.backward()
    epn.ep_sync_expert_grads(model)
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
        "ep": epn._mesh(model).size(),
        "has_bias_shards": sorted(
            {
                k.rsplit(".", 1)[-1]
                for k, p in model.named_parameters()
                if getattr(p, epn._EP, None) and "bias" in k
            }
        ),
    }


def _case_bypass(rank, world, tmp, repo):
    """A block forward that computes its experts without the experts call (a fused / decode-only path) raises."""
    from torch.distributed.device_mesh import init_device_mesh

    build, tpn, epn = _modules()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    model = _build(build, repo)
    epn.apply_expert_parallel(model, mesh)
    block = tpn.moe_blocks_with_experts(model)[0]
    ids = torch.randint(0, 50, (1, 4), generator = torch.Generator().manual_seed(0))
    with torch.no_grad():
        model(input_ids = ids)  # the normal path: no error
    block.forward = lambda hidden_states, *a, **k: torch.zeros_like(hidden_states)
    try:
        with torch.no_grad():
            model(input_ids = ids)
        message = None
    except RuntimeError as exc:
        message = str(exc)
    return {"message": message}


def _zoo_like_gpt_oss_decode(enable, last_expert = False):
    """GptOssMLP.forward with unsloth_zoo's shape: a 1-token eval decode reads the expert stacks directly
    (here: the experts class forward called unbound, around the experts call). last_expert routes every token
    to expert E - 1, a global id past this rank's E / ep local stacks."""
    from transformers.models.gpt_oss import modeling_gpt_oss as mod

    if not enable:
        return
    original = mod.GptOssMLP.forward

    def forward(self, hidden_states):
        bsz, qlen, hd = hidden_states.shape
        if qlen == 1 and not self.training:
            flat = hidden_states.reshape(-1, hd)
            _, scores, indices = self.router(flat)
            if last_expert:
                indices = torch.full_like(indices, self.router.num_experts - 1)
            out = type(self.experts).forward(self.experts, flat, indices, scores)
            return out.reshape(bsz, qlen, hd), scores
        return original(self, hidden_states)

    mod.GptOssMLP.forward = forward


def _case_generate(
    rank,
    world,
    tmp,
    repo,
    zoo_decode = False,
    plain_block = True,
    last_expert = False,
):
    _zoo_like_gpt_oss_decode(zoo_decode, last_expert = last_expert)
    """generate under expert parallel: each rank its own prompt and max_new_tokens (rank 1 stops 3 tokens early);
    synced_gpus keeps the finished rank stepping, and every rank's tokens equal one process' greedy decode."""
    from torch.distributed.device_mesh import init_device_mesh

    build, tpn, epn = _modules()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ref = _build(build, repo).eval()
    model = _build(build, repo).eval()
    if not plain_block:
        epn._BLOCK_FORWARDS.clear()
    epn.apply_expert_parallel(model, mesh)
    vocab = tpn._text_config(ref).vocab_size
    prompt = torch.randint(
        1, vocab, (1, 3 + rank), generator = torch.Generator().manual_seed(40 + rank)
    )
    kw = dict(max_new_tokens = 6 - 3 * rank, do_sample = False, pad_token_id = 0, eos_token_id = None)
    with torch.no_grad():
        want = ref.generate(input_ids = prompt, **kw)
        try:
            got = model.generate(input_ids = prompt, **kw)
        except RuntimeError as exc:
            return {"want": want, "got": None, "refused": str(exc)}
    return {"want": want, "got": got}


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
    results = _spawn("_case_family", repo = FAMILIES[family])
    _check_family(results, 2)
    if family == "gpt_oss":  # its per-expert biases are sharded with the stacks
        assert results[0]["has_bias_shards"] == ["down_proj_bias", "gate_up_proj_bias"], results[0][
            "has_bias_shards"
        ]


def test_ep_three_ranks_pad_experts_and_leave_one_rank_only_padding():
    # 4 experts over 3 ranks: 2 per rank, padded to 6, so rank 2 owns only zero experts the router never picks
    # (every call it receives no token: the empty-receiver path) and the splits are uneven three ways.
    _check_family(_spawn("_case_family", world = 3, repo = FAMILIES["olmoe"]), 3)


def _check_family(
    results,
    world,
    ep = None,
):
    ep = ep or world
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["ep"] == ep and r["blocks"] >= 1, r
        assert r["per_rank"] == max(2, -(-r["experts"] // ep)), r
        got, want = r["loss"]
        assert got == pytest.approx(want, rel = 1e-6), r["loss"]
        got, want = r["norm"]
        assert got == pytest.approx(want, rel = 1e-5), r["norm"]
        got, want = r["grads"]
        assert set(got) == set(want), sorted(set(got) ^ set(want))[:6]
        assert any(".experts." in k and "lora_" in k for k in want)
        for k in want:
            torch.testing.assert_close(got[k], want[k], rtol = 1e-5, atol = 1e-6, msg = k)


def test_ep_two_replica_groups_of_two_match_one_process():
    # ep=2 over 4 ranks: two groups each hold all 4 experts once (2 per rank); expert gradients are summed over
    # the groups. Each rank its own rows (4 different lengths).
    _check_family(_spawn("_case_family", world = 4, repo = FAMILIES["olmoe"], ep = 2), 4, ep = 2)


def test_ep_refuses_a_block_forward_that_skips_the_experts_call():
    results = _spawn("_case_bypass", repo = FAMILIES["olmoe"])
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["message"] is not None and "without calling" in r["message"], r["message"]


def test_ep_generate_with_uneven_lengths_matches_one_process():
    results = _spawn("_case_generate", repo = FAMILIES["qwen3_moe"])
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert torch.equal(r["got"], r["want"]), (r["got"], r["want"])
    assert results[0]["got"].shape[1] != results[1]["got"].shape[1]


def test_ep_gpt_oss_decode_path_around_the_experts_call():
    # zoo's GptOssMLP decodes 1 token in eval from the stacks directly: under expert parallel the block runs the
    # plain router -> experts forward instead, and decodes as one process does.
    results = _spawn("_case_generate", repo = FAMILIES["gpt_oss"], zoo_decode = True)
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["got"] is not None and torch.equal(r["got"], r["want"]), r
    # Without that replacement the bypass guard refuses rather than decoding from E / ep experts.
    results = _spawn("_case_generate", repo = FAMILIES["gpt_oss"], zoo_decode = True, plain_block = False)
    for r in results:
        assert r is not None and "without calling" in (r.get("refused") or ""), r


def test_ep_bypass_reading_a_global_expert_id_is_refused():
    # A decode path around the experts call that indexes this rank's E / ep local stacks with a global id >= E / ep
    # raised IndexError before the bypass guard's post hook (AMD gfx1151 hosts, torch 2.11: the router picked
    # expert 3 of 4 with 2 local). Every token routed to expert E - 1 makes that deterministic.
    results = _spawn(
        "_case_generate",
        repo = FAMILIES["gpt_oss"],
        zoo_decode = True,
        plain_block = False,
        last_expert = True,
    )
    for r in results:
        assert r is not None and "without calling" in (r.get("refused") or ""), r


@pytest.mark.parametrize("family", sorted(REFUSED))
def test_ep_refuses_models_it_cannot_shard(family):
    results = _spawn("_case_refused", repo = REFUSED[family])
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["message"] is not None and "no shardable MoE block" in r["message"], r["message"]
        assert r["unchanged"] and not r["is_ep"], r
