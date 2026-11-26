# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Expert parallel over per-expert ModuleList experts (the transformers 4.x MoE layout, the block loops over its
experts itself): gloo CPU, fp32, 2 or 3 ranks, each its own rows, vs one process on all of them. LoRA on every
Linear (attention, routers, shared experts and each routed expert). Loss, every gradient (the per-expert ones
gathered by name), the clipped norm, and the adapter save_pretrained writes on rank 0 (bitwise the one-process
adapter). Runs on the transformers in the environment; families whose experts are fused there are skipped."""

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

from test_ep_moe_families_gloo import FAMILIES, _local_config, _rows, _sum_loss  # noqa: E402

LIST_FAMILIES = (
    "mixtral",
    "olmoe",
    "qwen2_moe",
    "qwen3_moe",
    "glm4_moe",
    "deepseek_v3",
    "qwen3_next",
    "hunyuan_v1_moe",
)


def _modules():
    from test_tp_native_oracle import build, tpn
    from unsloth_models_loader import load_models_module
    return build, tpn, load_models_module("ep_native")


def _spec(family):
    spec = FAMILIES[family]
    return dict(spec) if isinstance(spec, dict) else dict(repo = spec)


def _build(build, repo, overrides, gc):
    model = build(repo, overrides = overrides).train()  # LoRA on all Linear layers
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():  # lora_B = 0 would leave every lora_A gradient of the first step at 0
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator = g) * 0.02)
    if gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
        model.enable_input_require_grads()
    return model


def _case_list_family(
    rank,
    world,
    tmp,
    repo = None,
    overrides = None,
    config = None,
    experts = None,
    gc = False,
):
    import torch.distributed as dist
    from peft import get_peft_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel

    build, tpn, epn = _modules()
    lists = epn._lists()
    if config is not None:
        repo = _local_config(tmp, rank, config)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ref = _build(build, repo, overrides, gc)
    blocks = tpn.moe_blocks_with_experts(ref)
    if not blocks or any(lists.list_shard_problem(b, world) is not None for b in blocks):
        return {
            "skip": f"no per-expert ModuleList experts here ({[lists.list_shard_problem(b, world) for b in blocks][:1]})"
        }
    vocab = tpn._text_config(ref).vocab_size
    rows = [_rows(vocab, r) for r in range(world)]
    n = sum(int((lab[:, 1:] != -100).sum()) for _, lab in rows)
    for ids, lab in rows:
        (_sum_loss(ref, ids, lab) / n).backward()
    want = {k: p.grad.detach().clone() for k, p in ref.named_parameters() if p.grad is not None}
    want_loss = sum(float(_sum_loss(ref, i, lab).detach()) for i, lab in rows) / n
    want_norm = float(
        torch.nn.utils.clip_grad_norm_([p for p in ref.parameters() if p.grad is not None], 1e9)
    )
    want_adapter = {k: v.detach().clone() for k, v in get_peft_model_state_dict(ref).items()}

    model = _build(build, repo, overrides, gc)
    epn.apply_expert_parallel(model, mesh)
    listed = lists.list_blocks(model)
    group = epn._mesh(model).get_group()
    ddp = DistributedDataParallel(model)
    loss = _sum_loss(ddp, *rows[rank]) * world / n
    loss.backward()
    epn.ep_sync_expert_grads(model)
    value = loss.detach() / world
    dist.all_reduce(value)
    mine = {
        k: p.grad.detach().clone()
        for k, p in model.named_parameters()
        if p.grad is not None and getattr(p, epn._EP, (None,))[0] == lists.LIST
    }
    parts = [None] * world
    dist.all_gather_object(parts, mine, group = group)
    got = {
        k: p.grad.detach().clone()
        for k, p in model.named_parameters()
        if p.grad is not None and k not in mine
    }
    for part in parts:
        got.update(part)
    norm = float(
        epn.ep_clip_grad_norm_(
            [p for p in model.parameters() if p.grad is not None], 1e9, group = group
        )
    )
    # The adapter as model.save_pretrained writes it: gathered on rank 0 only.
    out_dir = os.path.join(tmp, "adapter")
    model.save_pretrained(out_dir)
    dist.barrier()
    saved = None
    if rank == 0:
        from safetensors.torch import load_file
        saved = load_file(os.path.join(out_dir, "adapter_model.safetensors"))
    held = sorted(
        {
            int(k.split(".experts.")[1].split(".")[0])
            for k, _ in model.named_parameters()
            if ".experts." in k
        }
    )
    shard = listed[0].__dict__[lists._SHARD]
    return {
        "loss": (float(value), want_loss),
        "norm": (norm, want_norm),
        "grads": (got, want),
        "adapter": (saved, {k: v for k, v in want_adapter.items()}),
        "blocks": len(listed),
        "moe_blocks": len(blocks),
        "held": held,
        "range": (shard.lo, shard.hi, shard.experts),
        "calls": shard.calls,
    }


def _case_generate(
    rank,
    world,
    tmp,
    repo,
    overrides = None,
):
    """generate under expert parallel: each rank its own prompt and max_new_tokens (rank 1 stops 3 tokens early);
    synced_gpus keeps the finished rank stepping, every rank's tokens equal one process' greedy decode."""
    from torch.distributed.device_mesh import init_device_mesh

    build, tpn, epn = _modules()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ref = _build(build, repo, overrides, False).eval()
    model = _build(build, repo, overrides, False).eval()
    epn.apply_expert_parallel(model, mesh)
    if not epn._lists().list_blocks(model):
        return {"skip": "fused experts here"}
    vocab = tpn._text_config(ref).vocab_size
    prompt = torch.randint(
        1, vocab, (1, 3 + rank), generator = torch.Generator().manual_seed(40 + rank)
    )
    kw = dict(max_new_tokens = 6 - 3 * rank, do_sample = False, pad_token_id = 0, eos_token_id = None)
    with torch.no_grad():
        want = ref.generate(input_ids = prompt, **kw)
        got = model.generate(input_ids = prompt, **kw)
    return {"want": want, "got": got}


def _case_refuse_stacked_read(rank, world, tmp, repo):
    """A block forward that reads an expert's weights instead of calling it is refused on every rank."""
    from torch.distributed.device_mesh import init_device_mesh

    build, tpn, epn = _modules()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    model = _build(build, repo, None, False)
    block = tpn.moe_blocks_with_experts(model)[0]
    if epn._lists().list_shard_problem(block, world) is not None:
        return {"skip": "fused experts here"}
    epn.apply_expert_parallel(model, mesh)
    lists = epn._lists()
    inner = type(block).forward

    def reading(*a, **k):
        _ = block.experts[0].weight_of_a_fused_path  # a stacked fast path reading weights directly
        return inner(block, *a, **k)

    block.forward = lists._list_forward(block.__dict__[lists._SHARD], reading)
    ids, lab = _rows(tpn._text_config(model).vocab_size, rank)
    try:
        model(input_ids = ids, use_cache = False)
        return {"refused": None}
    except RuntimeError as exc:
        return {"refused": str(exc)}


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


def _check(results, world):
    assert all(r is not None and "error" not in r for r in results), results
    if any("skip" in r for r in results):
        pytest.skip(results[0]["skip"])
    held = []
    for r in results:
        assert r["blocks"] == r["moe_blocks"] >= 1, r
        assert (
            r["calls"] >= 1
        ), r  # the exchange ran (one count per block forward of the first block)
        got, want = r["loss"]
        assert got == pytest.approx(want, rel = 1e-6), r["loss"]
        got, want = r["norm"]
        assert got == pytest.approx(want, rel = 1e-5), r["norm"]
        got, want = r["grads"]
        assert set(got) == set(want), sorted(set(got) ^ set(want))[:6]
        assert any(".experts." in k and "lora_" in k for k in want)
        for k in want:
            torch.testing.assert_close(got[k], want[k], rtol = 1e-5, atol = 1e-6, msg = k)
        held.append(r["held"])
    lo, hi, E = results[0]["range"]
    assert sorted(e for h in held for e in h) == list(
        range(E)
    ), held  # each expert on exactly one rank
    saved, want = results[0]["adapter"]
    assert set(saved) == set(want), sorted(set(saved) ^ set(want))[:6]
    for k in want:
        assert torch.equal(saved[k], want[k]), k


@pytest.mark.parametrize("family", LIST_FAMILIES)
def test_ep_expert_lists_match_one_process(family):
    _check(_spawn("_case_list_family", **_spec(family)), 2)


@pytest.mark.parametrize("family", ("mixtral", "qwen2_moe"))
def test_ep_expert_lists_with_checkpointing_match_one_process(family):
    _check(_spawn("_case_list_family", gc = True, **_spec(family)), 2)


def test_ep_expert_lists_three_ranks_one_without_experts():
    # olmoe's tiny config: 4 experts over 3 ranks, 2 per rank, rank 2 holds none (receives nothing, still exchanges).
    _check(_spawn("_case_list_family", world = 3, **_spec("olmoe")), 3)


def test_ep_expert_lists_refuse_a_block_reading_expert_weights():
    results = _spawn("_case_refuse_stacked_read", repo = FAMILIES["mixtral"])
    assert all(r is not None and "error" not in r for r in results), results
    if any("skip" in r for r in results):
        pytest.skip(results[0]["skip"])
    for r in results:
        assert "read `weight_of_a_fused_path` of expert 0 directly" in (r["refused"] or ""), r


@pytest.mark.parametrize("family", ("mixtral", "deepseek_v3"))
def test_ep_expert_lists_generate_with_uneven_lengths_matches_one_process(family):
    results = _spawn("_case_generate", **_spec(family))
    assert all(r is not None and "error" not in r for r in results), results
    if any("skip" in r for r in results):
        pytest.skip(results[0]["skip"])
    for r in results:
        assert torch.equal(r["got"], r["want"]), r
