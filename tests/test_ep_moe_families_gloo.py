# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Expert parallel (gloo CPU, 2 ranks, fp32) on tiny MoE families vs one process on the concatenated batch.

Each rank trains its own rows (rank 1 a shorter sequence: uneven all-to-all splits) through torch DDP with the
routed experts sharded by index; LoRA on every Linear (the routers included) and on the fused expert stacks.
Models without routed experts, and experts kept as per-expert Linear lists (the load_in_4bit gpt-oss layout), are
refused before anything changes."""

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

FAMILIES = {
    "qwen3_moe": "trl-internal-testing/tiny-Qwen3MoeForCausalLM",
    "olmoe": "hf-internal-testing/tiny-random-OlmoeForCausalLM",
    "glm4_moe": "trl-internal-testing/tiny-Glm4MoeForCausalLM",  # shared experts + grouped routing
    # (hidden_states, router_indices, routing_weights), transposed stacks, per-expert biases
    "gpt_oss": "trl-internal-testing/tiny-GptOssForCausalLM",
    "mixtral": "hf-internal-testing/tiny-random-MixtralForCausalLM",
    "qwen2_moe": "hf-internal-testing/tiny-random-Qwen2MoeForCausalLM",
    "qwen3_next": "hf-internal-testing/tiny-random-Qwen3NextForCausalLM",
    # The tiny repo is all dense (first_k_dense_replace=3 of 2 layers).
    "deepseek_v3": dict(
        repo = "trl-internal-testing/tiny-DeepseekV3ForCausalLM",
        overrides = dict(
            first_k_dense_replace = 0,
            n_routed_experts = 8,
            n_group = 2,
            topk_group = 1,
            num_experts_per_tok = 2,
        ),
    ),
    "qwen3_5_moe": dict(
        config = dict(
            model_type = "qwen3_5_moe_text",
            vocab_size = 128,
            hidden_size = 32,
            num_hidden_layers = 2,
            layer_types = ["linear_attention", "full_attention"],
            num_attention_heads = 2,
            num_key_value_heads = 1,
            head_dim = 16,
            linear_num_key_heads = 2,
            linear_num_value_heads = 2,
            linear_key_head_dim = 8,
            linear_value_head_dim = 8,
            num_experts = 4,
            num_experts_per_tok = 2,
            moe_intermediate_size = 16,
            shared_expert_intermediate_size = 16,
        )
    ),
    # Experts directly on the decoder layer, beside a dense mlp.
    "gemma4": dict(
        config = dict(
            model_type = "gemma4_text",
            vocab_size = 128,
            vocab_size_per_layer_input = 128,
            hidden_size = 32,
            hidden_size_per_layer_input = 0,
            intermediate_size = 64,
            num_hidden_layers = 2,
            layer_types = ["sliding_attention", "full_attention"],
            sliding_window = 8,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            head_dim = 16,
            num_kv_shared_layers = 0,
            enable_moe_block = True,
            num_experts = 4,
            top_k_experts = 2,
            moe_intermediate_size = 16,
        ),
        experts = "experts",
    ),
    # Hunyuan-A13B layout (E from config, a top-k per layer, a shared expert), shrunk. Its default config has
    # num_experts = 1, which once put the family in the ep blocked-family table.
    "hunyuan_v1_moe": dict(
        config = dict(
            model_type = "hunyuan_v1_moe",
            vocab_size = 128,
            hidden_size = 32,
            intermediate_size = 16,
            num_hidden_layers = 3,
            num_attention_heads = 2,
            num_key_value_heads = 1,
            head_dim = 16,
            num_experts = 64,
            moe_topk = [8, 8, 8],
            num_shared_expert = [1, 1, 1],
            pad_token_id = 0,
            use_qk_norm = True,
        )
    ),
}
REFUSED = {
    "llama": "hf-internal-testing/tiny-random-LlamaForCausalLM",
}
EXPERT_LORA = dict(target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"])
_SEQ = (7, 5, 6, 4)


def _modules():
    from test_tp_native_oracle import build, tpn

    # Not test_tp_inkling2_gloo's loader: that file importorskips Inkling 2 at import, which on a
    # transformers without it raised Skipped in every rank and left these families with no result.
    from unsloth_models_loader import load_models_module
    return build, tpn, load_models_module("ep_native")


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


def _family(family):
    spec = FAMILIES[family]
    return dict(spec) if isinstance(spec, dict) else dict(repo = spec)


def _local_config(tmp, rank, config):
    from transformers import AutoConfig

    path = os.path.join(tmp, f"config_{rank}")
    config = dict(config)
    AutoConfig.for_model(config.pop("model_type"), **config).save_pretrained(path)
    return path


def _build(
    build,
    repo,
    overrides = None,
    experts = None,
):
    lora = EXPERT_LORA
    if experts is not None:
        lora = dict(target_parameters = [f"{experts}.gate_up_proj", f"{experts}.down_proj"])
    model = build(repo, overrides = overrides, **lora).train()
    g = torch.Generator().manual_seed(1234)
    with torch.no_grad():  # lora_B = 0 would leave every lora_A gradient of the first step at 0
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.copy_(torch.randn(p.shape, generator = g) * 0.02)
    return model


def _transformers4_gpt_oss_contract():
    """transformers 4.x gpt-oss on any version: dense [tokens, E] router scores, E read from their width."""
    import torch.nn.functional as F
    from transformers.models.gpt_oss import modeling_gpt_oss as mod

    def router_forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(hidden_states, self.weight, self.bias)
        top_value, indices = torch.topk(logits, self.top_k, dim = -1)
        top_value = torch.nn.functional.softmax(top_value, dim = 1, dtype = top_value.dtype)
        return torch.zeros_like(logits).scatter_(1, indices, top_value), indices

    def experts_forward(
        self,
        hidden_states,
        router_indices = None,
        routing_weights = None,
    ):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        num_experts = routing_weights.shape[1]
        next_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(router_indices, num_classes = num_experts + 1).permute(
                2, 1, 0
            )
            hit = torch.greater(mask.sum(dim = (-1, -2)), 0).nonzero()
        for expert in hit[:]:
            expert = expert[0]
            if expert == num_experts:
                continue
            with torch.no_grad():
                _, token = torch.where(mask[expert])
            gate_up = (
                hidden_states[token] @ self.gate_up_proj[expert] + self.gate_up_proj_bias[expert]
            )
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min = None, max = self.limit)
            up = up.clamp(min = -self.limit, max = self.limit)
            out = ((up + 1) * (gate * torch.sigmoid(gate * self.alpha))) @ self.down_proj[expert]
            out = out + self.down_proj_bias[expert]
            next_states.index_add_(0, token, out * routing_weights[token, expert, None])
        return next_states.view(batch_size, -1, self.hidden_size)

    def mlp_forward(self, hidden_states):
        scores, indices = self.router(hidden_states)
        out = self.experts(hidden_states, router_indices = indices, routing_weights = scores)
        return out.reshape(hidden_states.shape), scores

    mod.GptOssTopKRouter.forward = router_forward
    mod.GptOssExperts.forward = experts_forward
    mod.GptOssMLP.forward = mlp_forward


def _case_family(
    rank,
    world,
    tmp,
    repo = None,
    ep = None,
    overrides = None,
    config = None,
    experts = None,
    dense_routing = False,
):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.nn.parallel import DistributedDataParallel

    if dense_routing:
        _transformers4_gpt_oss_contract()
    build, tpn, epn = _modules()
    if config is not None:
        repo = _local_config(tmp, rank, config)
    ep = ep or world
    if ep == world:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    else:  # replica groups of ep ranks, ep innermost
        mesh = init_device_mesh("cpu", (world // ep, ep), mesh_dim_names = ("ep_rep", "ep"))
    ref = _build(build, repo, overrides, experts)
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

    model = _build(build, repo, overrides, experts)
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
            router_out = self.router(
                flat
            )  # transformers 5: (logits, scores, indices); 4.x: (scores, indices)
            scores, indices = router_out[-2:]
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


class _PerExpertListExperts(torch.nn.Module):
    """unsloth_zoo's GptOssExpertsBnb4bit layout (every load_in_4bit gpt-oss): one Linear per expert in
    gate_up_projs / down_projs, empty stack buffers, the gpt-oss experts call. bitsandbytes needs a GPU for real
    Linear4bit, and the layout alone is what expert parallel reads."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        hidden, inter = config.hidden_size, config.intermediate_size
        self.gate_up_projs = torch.nn.ModuleList(
            [torch.nn.Linear(hidden, 2 * inter) for _ in range(self.num_experts)]
        )
        self.down_projs = torch.nn.ModuleList(
            [torch.nn.Linear(inter, hidden) for _ in range(self.num_experts)]
        )
        for name in ("gate_up_proj", "gate_up_proj_bias", "down_proj", "down_proj_bias"):
            self.register_buffer(name, torch.empty(0), persistent = False)

    def forward(
        self,
        hidden_states,
        router_indices = None,
        routing_weights = None,
    ):
        raise AssertionError("never reached: the case only shards the model")


def _case_refused_per_expert_lists(rank, world, tmp, repo):
    from torch.distributed.device_mesh import init_device_mesh
    import transformers.models.gpt_oss.modeling_gpt_oss as gpt_oss

    build, tpn, epn = _modules()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    stock = gpt_oss.GptOssExperts
    gpt_oss.GptOssExperts = _PerExpertListExperts
    try:
        model = build(repo, lora = False)
    finally:
        gpt_oss.GptOssExperts = stock
    assert isinstance(model.model.layers[0].mlp.experts, _PerExpertListExperts)
    before = {k: (id(p), tuple(p.shape)) for k, p in model.named_parameters()}
    messages = []
    for call in (
        lambda: epn.ep_refuse_per_expert_lists(model),
        lambda: epn.apply_expert_parallel(model, mesh),
    ):
        try:
            call()
            messages.append(None)
        except NotImplementedError as exc:
            messages.append(str(exc))
    after = {k: (id(p), tuple(p.shape)) for k, p in model.named_parameters()}
    return {
        "messages": messages,
        "unchanged": before == after,
        "is_ep": epn.is_expert_parallel(model),
    }


_EXPORTS = (
    "save_pretrained_merged",
    "save_pretrained_gguf",
    "push_to_hub_merged",
    "push_to_hub_gguf",
)


def _case_save(rank, world, tmp, repo):
    """A direct save_pretrained of a LoRA model with unquantized experts: the adapter equals one process', and
    the gather receives only the expert LoRA factors, on rank 0 alone (never a frozen base stack). Then the merged
    / GGUF exports Unsloth binds on the model are refused without any Trainer (a custom training loop)."""
    import torch.distributed as dist
    from safetensors.torch import load_file
    from torch.distributed.device_mesh import init_device_mesh

    build, tpn, epn = _modules()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    ref = _build(build, repo)
    ref_dir = os.path.join(tmp, f"ref_{rank}")
    ref.save_pretrained(ref_dir)
    want = load_file(os.path.join(ref_dir, "adapter_model.safetensors"))
    experts = {k: p.numel() for k, p in ref.named_parameters() if ".experts." in k}
    base = {k: n for k, n in experts.items() if "lora_" not in k}
    lora = sum(n for k, n in experts.items() if "lora_" in k)

    model = _build(build, repo)
    calls = []
    for level in tpn._tp_chain(model):  # what patch_saving_functions binds on an Unsloth model
        for attr in _EXPORTS:
            setattr(level, attr, lambda *a, attr = attr, **k: calls.append(attr))
    epn.apply_expert_parallel(model, mesh)
    received = []
    all_gather, gather = dist.all_gather, dist.gather

    def counting_all_gather(tensor_list, tensor, *args, **kwargs):
        received.append(sum(t.numel() for t in tensor_list))
        return all_gather(tensor_list, tensor, *args, **kwargs)

    def counting_gather(
        tensor,
        gather_list = None,
        *args,
        **kwargs,
    ):
        received.append(sum(t.numel() for t in gather_list or []))
        return gather(tensor, gather_list, *args, **kwargs)

    dist.all_gather, dist.gather = counting_all_gather, counting_gather
    try:
        out = os.path.join(tmp, "ep")
        model.save_pretrained(out)
    finally:
        dist.all_gather, dist.gather = all_gather, gather
    got = load_file(os.path.join(out, "adapter_model.safetensors"))
    refused = {}
    for level in tpn._tp_chain(model):
        for attr in _EXPORTS:
            try:
                getattr(level, attr)("unused")
                refused[f"{type(level).__name__}.{attr}"] = None
            except NotImplementedError as exc:
                refused[f"{type(level).__name__}.{attr}"] = str(exc)
    return {
        "received": sum(received),
        "lora": lora,
        "smallest_base": min(base.values()),
        "worst": max(float((got[k] - want[k]).abs().max()) for k in want)
        if sorted(got) == sorted(want)
        else None,
        "expert_keys": sum(1 for k in got if ".experts." in k),
        "refused": refused,
        "calls": calls,
    }


def _case_lora_b_flip(rank, world, tmp, repo):
    """Rank state saved with lora_B cut rank_major, resumed under grouped_by_expert (same local shapes, other
    columns): refused, as is a checkpoint from before the layout was recorded; the matching layout loads."""
    from torch.distributed.device_mesh import init_device_mesh

    build, tpn, epn = _modules()
    assert epn._tpn() is tpn
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("ep",))["ep"]
    real = tpn._lora_b_layout

    def make(layout):
        tpn._lora_b_layout = lambda wrapper = None, adapter = None: layout
        os.environ["UNSLOTH_MOE_LORA_B_LAYOUT"] = layout
        try:
            return epn.apply_expert_parallel(_build(build, repo), mesh)
        finally:
            tpn._lora_b_layout = real
            os.environ.pop("UNSLOTH_MOE_LORA_B_LAYOUT", None)

    def load(model, directory):
        try:
            epn._load_rank_weights(model, directory)
            return None
        except ValueError as exc:
            return str(exc)

    saved = os.path.join(tmp, "ckpt")
    model = make("rank_major")
    epn._save_rank_state(model, None, saved)
    legacy = os.path.join(tmp, "legacy")
    path = epn._rank_path(legacy)
    os.makedirs(os.path.dirname(path), exist_ok = True)
    payload = torch.load(epn._rank_path(saved), weights_only = False)
    payload["layout"].pop("lora_b", None)
    torch.save(payload, path)
    flipped = make("grouped_by_expert")
    return {
        "recorded": sorted(set(epn._layout(model).get("lora_b", {}).values())),
        "same": load(make("rank_major"), saved),
        "flip": load(flipped, saved),
        "legacy_default": load(make("rank_major"), legacy),
        "legacy_flip": load(flipped, legacy),
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
    except BaseException as exc:  # pytest's Skipped / exit too: a rank must report, not vanish
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
    results = _spawn("_case_family", **_family(family))
    _check_family(results, 2)
    if family == "gpt_oss":  # its per-expert biases are sharded with the stacks
        assert results[0]["has_bias_shards"] == ["down_proj_bias", "gate_up_proj_bias"], results[0][
            "has_bias_shards"
        ]


def test_ep_gpt_oss_dense_routing_weights_match_one_process():
    _check_family(_spawn("_case_family", repo = FAMILIES["gpt_oss"], dense_routing = True), 2)


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


def test_ep_shards_per_expert_linear_lists():
    """The load_in_4bit gpt-oss experts layout (one experts call over per-expert Linear lists) is no longer refused:
    each rank keeps its experts of gate_up_projs / down_projs (parity in test_ep_expert_lists_gloo.py)."""
    results = _spawn("_case_refused_per_expert_lists", repo = FAMILIES["gpt_oss"])
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        assert r["messages"] == [None, None], r
        assert r["is_ep"] and not r["unchanged"], r


_SAVED = {}


def _saved():
    if "olmoe" not in _SAVED:
        _SAVED["olmoe"] = _spawn("_case_save", repo = FAMILIES["olmoe"])
    results = _SAVED["olmoe"]
    assert all(r is not None and "error" not in r for r in results), results
    return results


def test_ep_adapter_save_gathers_only_lora_factors_onto_rank0():
    rank0, rank1 = _saved()
    for r in (rank0, rank1):  # every rank reads the one file rank 0 wrote: the 1-GPU adapter
        assert r["worst"] == 0 and r["expert_keys"] > 0, r
    # No base expert stack is gathered anywhere, and only the writing rank receives anything.
    assert rank1["received"] == 0, rank1
    assert rank0["received"] == rank0["lora"] < rank0["smallest_base"], rank0


def test_ep_exports_refused_without_a_trainer():
    for r in _saved():
        assert not r["calls"], r["calls"]
        assert r["refused"] and all(
            m is not None and "not supported under expert parallel" in m
            for m in r["refused"].values()
        ), r["refused"]


def test_ep_resume_refuses_another_lora_b_layout():
    results = _spawn("_case_lora_b_flip", repo = FAMILIES["olmoe"])
    assert all(r is not None and "error" not in r for r in results), results
    for r in results:
        for key in ("flip", "legacy_flip"):
            assert r[key] is not None and "UNSLOTH_MOE_LORA_B_LAYOUT" in r[key], r
        assert r["same"] is None and r["legacy_default"] is None, r
        assert r["recorded"] == ["rank_major"], r
