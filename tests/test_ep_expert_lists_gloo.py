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
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))

from test_ep_moe_families_gloo import FAMILIES, _local_config, _rows, _sum_loss  # noqa: E402

_SHARED = ("shared_expert", "shared_experts", "shared_mlp")
SHARED_FAMILIES = ("qwen2_moe", "glm4_moe", "deepseek_v3", "qwen3_next", "hunyuan_v1_moe")

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


def _build(
    build,
    repo,
    overrides,
    gc,
    experts_and_router_only = False,
):
    model = build(repo, overrides = overrides).train()
    if experts_and_router_only:
        for name, p in model.named_parameters():
            p.requires_grad_("lora_" in name and (".experts." in name or ".gate." in name))
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
    experts_and_router_only = False,
    warm = False,
    shared_dropout = False,
):
    import torch.distributed as dist
    from peft import get_peft_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel

    build, tpn, epn = _modules()
    lists = epn._lists()
    shared_calls = {"ref": 0, "ep": 0}

    def count_shared(model, key):
        # `warm`: one no-grad forward first so pass 1 learns which children it may skip.
        if not warm:
            return
        with torch.no_grad():
            model(input_ids = rows[rank][0], use_cache = False)
        if (
            shared_dropout
        ):  # after the warm forward, as PEFT set_adapter to an adapter with LoRA dropout would
            for block in tpn.moe_blocks_with_experts(model):
                for name in _SHARED:
                    for m in getattr(block, name, torch.nn.Module()).modules():
                        for adapter in list(getattr(m, "lora_dropout", {}).keys()):
                            m.lora_dropout[adapter] = torch.nn.Dropout(0.0).train()
                            m.lora_dropout[
                                adapter
                            ].p = 1e-9  # active, draws randomness, numerically a no-op
        for block in tpn.moe_blocks_with_experts(model):
            for name in _SHARED:
                child = block._modules.get(name)
                if child is not None:
                    child.register_forward_hook(
                        lambda *a: shared_calls.__setitem__(key, shared_calls[key] + 1)
                    )

    if config is not None:
        repo = _local_config(tmp, rank, config)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ref = _build(build, repo, overrides, gc, experts_and_router_only)
    blocks = tpn.moe_blocks_with_experts(ref)
    if not blocks or any(lists.list_shard_problem(b, world) is not None for b in blocks):
        return {
            "skip": f"no per-expert ModuleList experts here ({[lists.list_shard_problem(b, world) for b in blocks][:1]})"
        }
    vocab = tpn._text_config(ref).vocab_size
    rows = [_rows(vocab, r) for r in range(world)]
    n = sum(int((lab[:, 1:] != -100).sum()) for _, lab in rows)
    count_shared(ref, "ref")
    for ids, lab in rows:
        (_sum_loss(ref, ids, lab) / n).backward()
    ref_calls = shared_calls["ref"]
    want = {k: p.grad.detach().clone() for k, p in ref.named_parameters() if p.grad is not None}
    want_loss = sum(float(_sum_loss(ref, i, lab).detach()) for i, lab in rows) / n
    want_norm = float(
        torch.nn.utils.clip_grad_norm_([p for p in ref.parameters() if p.grad is not None], 1e9)
    )
    want_adapter = {k: v.detach().clone() for k, v in get_peft_model_state_dict(ref).items()}

    model = _build(build, repo, overrides, gc, experts_and_router_only)
    epn.apply_expert_parallel(model, mesh)
    listed = lists.list_blocks(model)
    group = epn._mesh(model).get_group()
    count_shared(model, "ep")
    ddp = DistributedDataParallel(model)
    loss = _sum_loss(ddp, *rows[rank]) * world / n
    loss.backward()
    ep_calls = shared_calls["ep"]
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
        "shared_calls": (ep_calls, ref_calls / world),
        "pass1_skips": {k: [n for n, _ in v] for k, v in getattr(shard, "skip", {}).items()},
    }


def _case_portable(
    rank,
    world,
    tmp,
    repo,
    overrides = None,
):
    """Resume on another layout (portable_resume) with per-expert modules: the 1-GPU optimizer state gathered from
    an ep 2 step equals one process' after the same step, and a 1-GPU adapter loads into each rank's experts."""
    import torch.distributed as dist
    from peft import get_peft_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel
    from unsloth_models_loader import load_models_module

    build, tpn, epn = _modules()
    portable = load_models_module("portable_resume")
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ref = _build(build, repo, overrides, False)
    blocks = tpn.moe_blocks_with_experts(ref)
    if not blocks or epn._lists().list_shard_problem(blocks[0], world) is not None:
        return {"skip": "no per-expert ModuleList experts here"}
    vocab = tpn._text_config(ref).vocab_size
    rows = [_rows(vocab, r) for r in range(world)]
    n = sum(int((lab[:, 1:] != -100).sum()) for _, lab in rows)

    def adam(model):
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr = 1e-2, foreach = False
        )

    ref_opt = adam(ref)
    for ids, lab in rows:
        (_sum_loss(ref, ids, lab) / n).backward()
    ref_opt.step()
    names = {id(p): k for k, p in ref.named_parameters()}
    want = {names[id(p)]: {k: v.clone() for k, v in st.items()} for p, st in ref_opt.state.items()}
    model = _build(build, repo, overrides, False)
    epn.apply_expert_parallel(model, mesh)
    opt = adam(model)
    ddp = DistributedDataParallel(model)
    (_sum_loss(ddp, *rows[rank]) * world / n).backward()
    epn.ep_sync_expert_grads(model)
    opt.step()
    state, why = portable.portable_optimizer_state(model, opt)
    worst, missing = 0.0, []
    if rank == 0:
        got = state["state"]
        missing = sorted(set(want) ^ set(got))
        for k in set(want) & set(got):
            for key in ("exp_avg", "exp_avg_sq"):
                worst = max(worst, float((got[k][key] - want[k][key]).abs().max()))
    ckpt = os.path.join(tmp, "ckpt")
    if rank == 0:
        ref.save_pretrained(ckpt)
    dist.barrier()
    fresh = _build(build, repo, overrides, False)
    epn.apply_expert_parallel(fresh, mesh)
    portable.portable_load_weights(fresh, ckpt)
    ref_sd = {k: v for k, v in ref.named_parameters()}
    load_worst = max(
        float((p.detach() - ref_sd[k].detach()).abs().max())
        for k, p in fresh.named_parameters()
        if p.requires_grad and "lora_" in k
    )
    experts_loaded = sum(
        1 for k, p in fresh.named_parameters() if ".experts." in k and "lora_" in k
    )
    order = [names[id(p)] for p in ref_opt.param_groups[0]["params"]]
    portable_order = state["param_groups"][0]["params"] if rank == 0 else None
    fresh_opt = adam(fresh)
    portable.portable_load_optimizer(fresh, fresh_opt, ref_opt.state_dict())
    fresh_names = {id(p): k for k, p in fresh.named_parameters()}
    positional_worst = max(
        float((st["exp_avg"] - want[fresh_names[id(p)]]["exp_avg"]).abs().max())
        for p, st in fresh_opt.state.items()
    )
    return {
        "why": why,
        "missing": missing,
        "worst": worst,
        "load_worst": load_worst,
        "experts_loaded": experts_loaded,
        "order_ok": portable_order is None or portable_order == order,
        "positional_worst": positional_worst,
        "positional_n": len(fresh_opt.state),
    }


class _RandomRouterMoE(torch.nn.Module):
    """A block that loops its experts, with a sampling router (noise drawn every forward) and dropout inside each
    expert, recording the router's draws: both passes of a forward must draw the same noise."""

    def __init__(
        self,
        E = 4,
        H = 8,
    ):
        super().__init__()
        self.gate = torch.nn.Linear(H, E)
        self.experts = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(H, H), torch.nn.Dropout(0.5)) for _ in range(E)
        )
        self.draws = []

    def forward(self, x):
        noise = torch.rand(x.shape[0], 1)
        self.draws.append(noise.clone())
        top = self.gate(x * (1 + noise)).softmax(-1).topk(2, dim = -1)
        out = torch.zeros_like(x)
        for e in range(len(self.experts)):
            token, slot = torch.where(top.indices == e)
            if token.numel():
                out = out.index_add(
                    0, token, self.experts[e](x[token]) * top.values[token, slot, None]
                )
        return out


def _case_random_router(rank, world, tmp):
    from torch.distributed.device_mesh import init_device_mesh

    _, tpn, epn = _modules()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    torch.manual_seed(0)
    model = torch.nn.Sequential(_RandomRouterMoE()).train()
    epn.apply_expert_parallel(model, mesh)
    block = model[0]
    torch.manual_seed(10 + rank)
    same = []
    for step in range(3):
        block.draws.clear()
        model(torch.randn(6, 8)).sum().backward()
        # Step 0 adds one no-grad pass 1 (skip learning).
        same.append(
            len(block.draws) == (3 if step == 0 else 2)
            and all(torch.equal(block.draws[0], d) for d in block.draws)
        )
    return {"same": same}


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


def _gpt_oss_linear_lists(gc = False):
    """A tiny gpt-oss whose experts are unsloth_zoo's GptOssExpertsBnb4bit (the load_in_4bit layout: one fused
    experts call over per-expert Linear lists), here with plain fp32 Linears; LoRA on every Linear."""
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    os.environ.setdefault("UNSLOTH_FORCE_CPU", "1")
    import transformers
    from peft import LoraConfig, get_peft_model

    try:
        from unsloth_zoo.temporary_patches.gpt_oss import GptOssExpertsBnb4bit, GptOssTopKRouter
    except Exception as exc:
        return None, f"unsloth_zoo GptOssExpertsBnb4bit unavailable: {type(exc).__name__}: {exc}"[
            :400
        ]
    config = transformers.GptOssConfig(
        vocab_size = 128,
        hidden_size = 32,
        intermediate_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        num_local_experts = 4,
        num_experts_per_tok = 2,
        sliding_window = 8,
        layer_types = ["sliding_attention", "full_attention"],
        tie_word_embeddings = False,
    )
    config.dtype = torch.float32
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, dtype = torch.float32)
    g = torch.Generator().manual_seed(7)
    for layer in model.model.layers:
        experts = GptOssExpertsBnb4bit(config)
        with torch.no_grad():
            for p in experts.parameters():
                p.copy_(torch.randn(p.shape, generator = g) * 0.1)
        layer.mlp.experts = experts
        # The zoo's router too: its dense [tokens, E] scores are what GptOssExpertsBnb4bit indexes (transformers
        # 5's own router returns [tokens, top_k]).
        router = GptOssTopKRouter(config)
        with torch.no_grad():
            router.weight.copy_(layer.mlp.router.weight)
            router.bias.copy_(layer.mlp.router.bias)
        layer.mlp.router = router
    model = get_peft_model(
        model, LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)
    )
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator = g) * 0.02)
    if gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs = {"use_reentrant": False})
        model.enable_input_require_grads()
    return model.train(), None


def _case_fused_lists(
    rank,
    world,
    tmp,
    gc = False,
):
    import torch.distributed as dist
    from peft import get_peft_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel

    _, tpn, epn = _modules()
    lists = epn._lists()
    ref, why = _gpt_oss_linear_lists(gc)
    if ref is None:
        return {"skip": why}
    block = tpn.moe_blocks_with_experts(ref)[0]
    assert tpn.expert_shard_problem(block, world).startswith(
        "per-expert Linear lists"
    ), tpn.expert_shard_problem(block, world)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    rows = [_rows(128, r) for r in range(world)]
    n = sum(int((lab[:, 1:] != -100).sum()) for _, lab in rows)
    for ids, lab in rows:
        (_sum_loss(ref, ids, lab) / n).backward()
    want = {k: p.grad.detach().clone() for k, p in ref.named_parameters() if p.grad is not None}
    want_loss = sum(float(_sum_loss(ref, i, lab).detach()) for i, lab in rows) / n
    want_norm = float(
        torch.nn.utils.clip_grad_norm_([p for p in ref.parameters() if p.grad is not None], 1e9)
    )
    want_adapter = {k: v.detach().clone() for k, v in get_peft_model_state_dict(ref).items()}
    model, _ = _gpt_oss_linear_lists(gc)
    epn.ep_refuse_per_expert_lists(model)
    epn.apply_expert_parallel(model, mesh)
    group = epn._mesh(model).get_group()
    ddp = DistributedDataParallel(model)
    loss = _sum_loss(ddp, *rows[rank]) * world / n
    loss.backward()
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
    out_dir = os.path.join(tmp, "adapter")
    model.save_pretrained(out_dir)
    dist.barrier()
    saved = None
    if rank == 0:
        from safetensors.torch import load_file
        saved = load_file(os.path.join(out_dir, "adapter_model.safetensors"))
    held = sorted(
        {
            int(k.split("_projs.")[1].split(".")[0])
            for k, _ in model.named_parameters()
            if "_projs." in k
        }
    )
    exchange = lists.list_blocks(model)[0].__dict__["_unsloth_tp_expert_shard"]
    return {
        "loss": (float(value), want_loss),
        "norm": (norm, want_norm),
        "grads": (got, want),
        "adapter": (saved, want_adapter),
        "blocks": len(lists.list_blocks(model)),
        "moe_blocks": len(tpn.moe_blocks_with_experts(model)),
        "held": held,
        "range": (exchange.lo, exchange.hi, exchange.experts),
        "calls": exchange.calls,
    }


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
        _ = block.experts[0].weight_of_a_fused_path
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
    except BaseException as exc:
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
        assert r["calls"] >= 1, r
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
    assert sorted(e for h in held for e in h) == list(range(E)), held
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


@pytest.mark.parametrize("family", SHARED_FAMILIES)
@pytest.mark.parametrize("gc", (False, True))
def test_ep_expert_lists_run_the_shared_expert_once(family, gc):
    """After the first forward, pass 1 skips the shared expert: calls and results match one process."""
    if gc and family not in ("qwen2_moe", "deepseek_v3"):
        pytest.skip("checkpointing: two families are enough")
    results = _spawn("_case_list_family", gc = gc, warm = True, **_spec(family))
    _check(results, 2)
    for r in results:
        got, want = r["shared_calls"]
        assert want > 0 and got == want, r["shared_calls"]
        assert any(r["pass1_skips"].values()), r["pass1_skips"]


def test_ep_expert_lists_run_the_shared_expert_in_pass1_while_its_dropout_draws():
    """A shared expert dropout enabled after warmup still runs in pass 1, keeping the RNG in sync."""
    results = _spawn("_case_list_family", warm = True, shared_dropout = True, **_spec("qwen2_moe"))
    _check(results, 2)
    for r in results:
        got, want = r["shared_calls"]
        assert want > 0 and got == 2 * want, r["shared_calls"]


def test_ep_expert_lists_watch_only_shared_expert_children():
    """A router as wide as the hidden size is H to H too, but only the shared expert children are candidates."""
    from unsloth.models import ep_expert_lists as lists

    block = torch.nn.Module()
    block.gate = torch.nn.Linear(4, 4)
    block.shared_expert = torch.nn.Linear(4, 4)
    block.experts = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
    with lists._watch_shapes(block) as found:
        x = torch.randn(3, 4)
        block.gate(x), block.shared_expert(x), block.experts[0](x)
    assert found == {"shared_expert": torch.float32}, found


def test_ep_expert_lists_three_ranks_one_without_experts():
    # 4 experts over 3 ranks: rank 2 holds none but still exchanges.
    _check(_spawn("_case_list_family", world = 3, **_spec("olmoe")), 3)


def test_ep_expert_lists_three_ranks_only_experts_and_routers_train():
    _check(_spawn("_case_list_family", world = 3, experts_and_router_only = True, **_spec("olmoe")), 3)


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


@pytest.mark.parametrize("gc", (False, True))
def test_ep_gpt_oss_linear_lists_match_one_process(gc):
    _check(_spawn("_case_fused_lists", gc = gc), 2)


def test_ep_expert_lists_portable_resume_matches_one_process():
    results = _spawn("_case_portable", **_spec("mixtral"))
    assert all(r is not None and "error" not in r for r in results), results
    if any("skip" in r for r in results):
        pytest.skip(results[0]["skip"])
    assert results[0]["why"] is None and not results[0]["missing"], results[0]
    assert results[0]["worst"] <= 1e-6, results[0]
    for r in results:
        assert r["experts_loaded"] > 0 and r["load_worst"] == 0.0, r
        assert r["order_ok"], r
        assert r["positional_n"] > 0 and r["positional_worst"] == 0.0, r


def test_ep_expert_lists_both_passes_draw_the_same_router_randomness():
    results = _spawn("_case_random_router")
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["same"] == [True, True, True], r


def _case_zoo_fast_paths(rank, world, tmp, repo):
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    os.environ.setdefault("UNSLOTH_FORCE_CPU", "1")
    build, tpn, epn = _modules()
    lists = epn._lists()
    try:
        from unsloth_zoo.temporary_patches import moe_grouped_modulelist as zoo
    except Exception as exc:
        return {"skip": f"unsloth_zoo moe_grouped_modulelist unavailable: {type(exc).__name__}"}
    model = _build(build, repo, None, False)
    block = tpn.moe_blocks_with_experts(model)[0]
    spec = zoo._BLOCK_SPECS.get(type(block).__name__)
    if type(block._modules.get("experts")) is not torch.nn.ModuleList:
        return {"skip": "no per-expert ModuleList experts here"}
    if spec is None:
        return {"skip": f"no zoo grouped spec for {type(block).__name__}"}
    before = lists.list_shard_problem(block, world)
    block._orig_moe_forward = block.forward
    grouped = lists.list_shard_problem(block, world)
    del block._orig_moe_forward
    stacked = zoo._stack_block_lora(block, spec)
    return {
        "before": before,
        "grouped": grouped,
        "stacked": stacked,
        "after": lists.list_shard_problem(block, world),
    }


def test_ep_expert_lists_refuse_the_zoo_stacked_lora_and_grouped_forward():
    out = _spawn("_case_zoo_fast_paths", world = 1, repo = FAMILIES["mixtral"])[0]
    assert out is not None and "error" not in out, out
    if "skip" in out:
        pytest.skip(out["skip"])
    assert out["before"] is None, out
    assert "UNSLOTH_MOE_GROUPED=0" in out["grouped"], out
    assert out["stacked"] >= 1, out
    assert "UNSLOTH_MOE_STACKED_LORA=0" in out["after"], out


def test_ep_env_load_keeps_the_zoo_per_expert_paths(monkeypatch):
    _, _, epn = _modules()
    monkeypatch.delenv("UNSLOTH_MOE_STACKED_LORA", raising = False)
    monkeypatch.setenv("UNSLOTH_MOE_GROUPED", "1")
    assert epn.ep_env_load(lambda: "loaded", (), {}, False) == "loaded"
    assert os.environ["UNSLOTH_MOE_STACKED_LORA"] == "0"
    assert os.environ["UNSLOTH_MOE_GROUPED"] == "1"
