# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ulysses context parallelism for Kimi-Linear / Kimi K3 (KDA + MLA), on CPU gloo ranks.

Kimi ships remote modeling code, so the layers here are stand-ins with the remote class names and
attribute layout, and a plain full-sequence torch reference forward.
"""

import os
import tempfile
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

import unsloth  # noqa: F401
from unsloth import context_parallel as cp
from unsloth.utils import attention_dispatch as ad
from unsloth.utils import ulysses_hybrid as uh

transformers = pytest.importorskip("transformers")
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

HIDDEN, HEADS, HEAD_DIM, VOCAB = 32, 4, 8, 64


# moonshotai/Kimi-Linear-48B-A3B-Instruct modeling_kimi.py, verbatim: the MoE MLPs, the gate that adds its score
# bias in place to the sigmoid output, and the inference-only sparse block (Unsloth's remote MoE shim trains it).
_HUB_KIMI_MOE_SRC = r'''
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN


class KimiBlockSparseMLP(nn.Module):
    def __init__(self, config: KimiLinearConfig, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.ffn_dim = config.intermediate_size if intermediate_size is None else intermediate_size
        self.hidden_dim = config.hidden_size if hidden_size is None else hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)   # gate
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)   # down
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)   # up

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(
            self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class KimiMLP(nn.Module):
    def __init__(self, config: KimiLinearConfig, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(
            self.gate_proj(x)) * self.up_proj(x))
        return down_proj



class KimiMoEGate(nn.Module):
    """
    MoEGate adapted from Deepseek-V3.
    Parameter correspondences:
        num_experts -> n_routed_experts
        num_experts_per_token -> num_experts_per_tok
        num_expert_group -> n_group
        moe_router_activation_func -> scoring_func
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_token
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_router_activation_func = config.moe_router_activation_func
        self.num_expert_group = getattr(config, "num_expert_group", 1)
        self.topk_group = getattr(config, "topk_group", 1)

        # topk selection algorithm
        self.moe_renormalize = config.moe_renormalize
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, self.gating_dim)),
        )

        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(
            hidden_states.type(torch.float32), self.weight.type(
                torch.float32), None,
        )
        if self.moe_router_activation_func == "sigmoid":
            scores = logits.sigmoid()
        elif self.moe_router_activation_func == "softmax":
            scores = logits.softmax(dim=1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.moe_router_activation_func}",
            )

        # select top-k experts
        assert not self.training
        scores_for_choice = scores.view(bsz * seq_len, -1)
        scores_for_choice += self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(
                bsz * seq_len, self.num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )  # [n, num_expert_group]
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False,
        )[
            1
        ]  # [n, top_k_group]
        group_mask = torch.zeros_like(group_scores)  # [n, num_expert_group]
        group_mask.scatter_(1, group_idx, 1)  # [n, num_expert_group]
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(
                bsz * seq_len, self.num_expert_group, self.num_experts // self.num_expert_group,
            )
            .reshape(bsz * seq_len, -1)
        )  # [n, e]
        tmp_scores = scores_for_choice.masked_fill(
            ~score_mask.bool(), 0.0)  # [n, e]
        _, topk_idx = torch.topk(
            tmp_scores, k=self.top_k, dim=-1, sorted=False,
        )
        topk_weight = scores.gather(1, topk_idx)

        # norm gate to sum 1
        if self.top_k > 1 and self.moe_renormalize:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        # must multiply the scaling factor
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight



class KimiSparseMoeBlock(nn.Module):
    """
    Adapted from Deepseek-V3's MOE implementation
    The namings are consistent with Kimi's version.
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.moe_renormalize = config.moe_renormalize

        self.ep_size = 1
        self.experts_per_rank = config.num_experts
        self.ep_rank = 0
        self.experts = nn.ModuleList(
            [
                KimiBlockSparseMLP(
                    config, intermediate_size=config.moe_intermediate_size,
                )
                for _ in range(config.num_experts)
            ],
        )
        self.gate = KimiMoEGate(config)
        if config.num_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = KimiMLP(
                config=config, intermediate_size=intermediate_size,
            )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        else:
            raise NotImplementedError("Training mode is not supported in KimiSparseMoeBlock")
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]

        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(
            outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out

'''
_MOE_CONFIG = types.SimpleNamespace(
    hidden_size = HIDDEN,
    intermediate_size = 2 * HIDDEN,
    moe_intermediate_size = 16,
    hidden_act = "silu",
    num_experts = 8,
    num_experts_per_token = 2,
    routed_scaling_factor = 2.446,
    moe_router_activation_func = "sigmoid",
    num_expert_group = 1,
    topk_group = 1,
    moe_renormalize = True,
    num_shared_experts = 1,
)


def _hub_kimi_moe_module(name = "transformers_modules.tiny_kimi_cp.modeling_kimi"):
    import linecache
    import sys

    if name in sys.modules:
        return sys.modules[name]
    filename = f"<{name}>"
    linecache.cache[filename] = (
        len(_HUB_KIMI_MOE_SRC),
        None,
        _HUB_KIMI_MOE_SRC.splitlines(True),
        filename,
    )
    module = types.ModuleType(name)
    module.__dict__["__name__"] = name
    exec(compile(_HUB_KIMI_MOE_SRC, filename, "exec"), module.__dict__)
    sys.modules[name] = module
    return module


class _Config(types.SimpleNamespace):
    # Stands in for a PretrainedConfig: PEFT 0.18 reads model.config.to_dict().
    def to_dict(self):
        return dict(vars(self))


def _acc(x):
    # fp32 math for low precision, fp64 kept for the fp64 reference.
    return torch.promote_types(x.dtype, torch.float32)


class _ShortConv(nn.Conv1d):
    # fla.modules.ShortConvolution: depthwise causal conv with a SiLU.
    def __init__(
        self,
        channels,
        kernel = 4,
    ):
        super().__init__(
            channels, channels, kernel, groups = channels, bias = False, padding = kernel - 1
        )
        self.activation = "silu"

    def forward(self, x):
        return F.silu(super().forward(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2))


class _GatedNorm(nn.Module):
    # fla FusedRMSNormGated(activation = "sigmoid").
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim) + 0.1 * torch.randn(dim))

    def forward(self, x, g):
        x = x.to(_acc(x))
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + 1e-6)
        return (self.weight * x * torch.sigmoid(g.to(_acc(g)))).to(g.dtype)


class KimiDeltaAttention(nn.Module):
    def __init__(self, k3):
        super().__init__()
        proj = HEADS * HEAD_DIM
        self.num_heads, self.head_dim = HEADS, HEAD_DIM
        self.q_proj, self.k_proj, self.v_proj = (
            nn.Linear(HIDDEN, proj, bias = False) for _ in range(3)
        )
        self.q_conv1d, self.k_conv1d, self.v_conv1d = (_ShortConv(proj) for _ in range(3))
        self.A_log = nn.Parameter(0.5 * torch.randn(HEADS))
        self.dt_bias = nn.Parameter(0.5 * torch.randn(proj))
        self.f_a_proj = nn.Linear(HIDDEN, HEAD_DIM, bias = False)
        self.f_b_proj = nn.Linear(HEAD_DIM, proj, bias = False)
        self.b_proj = nn.Linear(HIDDEN, HEADS, bias = False)
        self.gate_lower_bound = -5.0 if k3 else None
        if k3:  # use_full_rank_gate
            self.g_proj = nn.Linear(HIDDEN, proj, bias = False)
        else:
            self.g_a_proj = nn.Linear(HIDDEN, HEAD_DIM, bias = False)
            self.g_b_proj = nn.Linear(HEAD_DIM, proj, bias = False)
        self.o_norm = _GatedNorm(HEAD_DIM)
        self.o_proj = nn.Linear(proj, HIDDEN, bias = False)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        cache_params = None,
        **kwargs,
    ):
        bsz, seq, _ = hidden_states.shape
        shape = (bsz, seq, HEADS, HEAD_DIM)
        q = self.q_conv1d(self.q_proj(hidden_states)).view(shape)
        k = self.k_conv1d(self.k_proj(hidden_states)).view(shape)
        v = self.v_conv1d(self.v_proj(hidden_states)).view(shape)
        acc = _acc(hidden_states)
        f = self.f_b_proj(self.f_a_proj(hidden_states)).view(shape).to(acc) + self.dt_bias.view(
            HEADS, HEAD_DIM
        )
        a = self.A_log.to(acc).exp().view(HEADS, 1)
        if self.gate_lower_bound is not None:
            g = self.gate_lower_bound * torch.sigmoid(a * f)
        else:
            g = -a * F.softplus(f)
        beta = self.b_proj(hidden_states).to(acc).sigmoid()
        # fla l2norm (use_qk_l2norm_in_kernel): eps 1e-6 under the square root.
        q, k = q.to(acc), k.to(acc)
        q = q * torch.rsqrt(q.pow(2).sum(-1, keepdim = True) + 1e-6) * HEAD_DIM**-0.5
        k = k * torch.rsqrt(k.pow(2).sum(-1, keepdim = True) + 1e-6)
        state = q.new_zeros(bsz, HEADS, HEAD_DIM, HEAD_DIM)
        outs = []
        for t in range(seq):
            state = state * g[:, t].exp().unsqueeze(-1)
            pred = (k[:, t].unsqueeze(-1) * state).sum(-2)
            state = state + k[:, t].unsqueeze(-1) * (
                beta[:, t, :, None] * (v[:, t].to(acc) - pred)
            ).unsqueeze(-2)
            outs.append((q[:, t].unsqueeze(-1) * state).sum(-2))
        o = torch.stack(outs, dim = 1).to(hidden_states.dtype)
        if hasattr(self, "g_proj"):
            gate = self.g_proj(hidden_states)
        else:
            gate = self.g_b_proj(self.g_a_proj(hidden_states))
        o = self.o_norm(o, gate.view(shape))
        o = self.o_proj(o.reshape(bsz, seq, -1))
        if attention_mask is not None:
            # The hub layer unpads on the 2D mask and pad_input leaves zero rows at the padding; with
            # right padding the causal conv and scan never carry it into the real tokens.
            o = o * attention_mask[:, -seq:, None].to(o.dtype)
        return o


def eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout = 0.0,
    **kwargs,
):
    # The hub modeling file's own eager attention (Ulysses swaps this module global).
    weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask[:, :, :, : key.shape[-2]]
    weights = F.softmax(weights, dim = -1, dtype = _acc(query)).to(query.dtype)
    return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights


def _causal_mask(
    length,
    dtype,
    padding = None,
):
    mask = torch.full((length, length), torch.finfo(dtype).min, dtype = dtype).triu(1)[None, None]
    if padding is not None:
        # As transformers' create_causal_mask (sdpa / eager): padded keys are masked for every query,
        # so a padding query attends its row's real keys.
        mask = mask.masked_fill(padding[:, None, None, :] == 0, torch.finfo(dtype).min)
    return mask


class KimiMLAAttention(nn.Module):
    # No RoPE (mla_use_nope); query / key heads 12 wide, value heads 8.
    def __init__(self, config, k3):
        super().__init__()
        self.config = config
        self.num_heads, self.num_key_value_groups, self.is_causal = HEADS, 1, True
        self.qk_nope, self.qk_rope, self.v_dim, self.kv_lora = 8, 4, 8, 16
        self.q_dim = self.qk_nope + self.qk_rope
        self.scaling = self.q_dim**-0.5
        self.q_proj = nn.Linear(HIDDEN, HEADS * self.q_dim, bias = False)
        self.kv_a_proj_with_mqa = nn.Linear(HIDDEN, self.kv_lora + self.qk_rope, bias = False)
        self.kv_b_proj = nn.Linear(self.kv_lora, HEADS * (self.qk_nope + self.v_dim), bias = False)
        self.o_proj = nn.Linear(HEADS * self.v_dim, HIDDEN, bias = False)
        self.use_output_gate = k3  # mla_use_output_gate
        if k3:
            self.g_proj = nn.Linear(HIDDEN, HEADS * self.v_dim, bias = False)

    def forward(
        self,
        hidden_states,
        attention_mask = None,
        **kwargs,
    ):
        bsz, seq, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seq, HEADS, self.q_dim).transpose(1, 2)
        kv, k_rot = self.kv_a_proj_with_mqa(hidden_states).split(
            [self.kv_lora, self.qk_rope], dim = -1
        )
        kv = self.kv_b_proj(kv).view(bsz, seq, HEADS, -1).transpose(1, 2)
        k_nope, value = kv.split([self.qk_nope, self.v_dim], dim = -1)
        k_rot = k_rot.view(bsz, 1, seq, self.qk_rope).expand(-1, HEADS, -1, -1)
        key = torch.cat([k_nope, k_rot], dim = -1)
        implementation = self.config._attn_implementation
        attention = (
            eager_attention_forward
            if implementation == "eager"
            else ALL_ATTENTION_FUNCTIONS[implementation]
        )
        out, _ = attention(self, q, key, value, attention_mask, dropout = 0.0, scaling = self.scaling)
        out = out.reshape(bsz, seq, -1)
        if self.use_output_gate:
            out = out * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(out)


class _Layer(nn.Module):
    def __init__(
        self,
        config,
        linear,
        k3,
        moe = None,
    ):
        super().__init__()
        self.self_attn = KimiDeltaAttention(k3) if linear else KimiMLAAttention(config, k3)
        if moe is not None:
            self.block_sparse_moe = moe  # the hub KimiSparseMoeBlock
        else:
            self.mlp = nn.Sequential(
                nn.Linear(HIDDEN, 2 * HIDDEN), nn.SiLU(), nn.Linear(2 * HIDDEN, HIDDEN)
            )
        self.norm1, self.norm2 = nn.LayerNorm(HIDDEN), nn.LayerNorm(HIDDEN)

    def forward(
        self,
        h,
        attention_mask = None,
    ):
        # Kimi hands the delta layers the 2D padding mask and MLA the causal mask.
        if isinstance(self.self_attn, KimiDeltaAttention):
            mask = attention_mask
        else:
            mask = _causal_mask(h.shape[1], h.dtype, attention_mask)
        h = h + self.self_attn(hidden_states = self.norm1(h), attention_mask = mask)
        mlp = getattr(self, "block_sparse_moe", None) or self.mlp
        return h + mlp(self.norm2(h))


class _KimiForCausalLM(nn.Module):
    def __init__(
        self,
        k3,
        moe = False,
        attn = "sdpa",
        gc = False,
    ):
        super().__init__()
        self.gc = gc
        self.config = _Config(
            _attn_implementation = attn,
            num_attention_heads = HEADS,
            num_key_value_heads = HEADS,
            output_router_logits = False,
            vision_config = None,
        )
        blocks = [None] * 3
        if moe:
            hub = _hub_kimi_moe_module()
            moe_config = types.SimpleNamespace(**vars(_MOE_CONFIG))
            if moe is not True:
                moe_config.num_experts = moe
            self.config.model_type = "kimi_linear"
            self.config.num_experts = moe_config.num_experts
            blocks = [hub.KimiSparseMoeBlock(moe_config) for _ in blocks]
            for block in blocks:
                with torch.no_grad():
                    block.gate.e_score_correction_bias.normal_(0, 0.3)  # nonzero, as trained
                    block.gate.e_score_correction_bias[-2:] = -5.0  # two experts no token picks
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            _Layer(self.config, linear, k3, block)
            for linear, block in zip((True, False, True), blocks)
        )
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias = False)

    def forward(
        self,
        input_ids,
        position_ids = None,
        use_cache = False,
        attention_mask = None,
        **kwargs,
    ):
        h = self.embed(input_ids)
        for layer in self.model.layers:
            if self.gc:
                # As HF gradient checkpointing runs under DDP: non-reentrant, recomputed in backward.
                from torch.utils.checkpoint import checkpoint
                h = checkpoint(layer, h, attention_mask, use_reentrant = False)
            else:
                h = layer(h, attention_mask)
        return types.SimpleNamespace(logits = self.lm_head(h))


def _model(k3, **kwargs):
    torch.manual_seed(0)
    model = _KimiForCausalLM(k3, **kwargs).train()
    if kwargs.get("moe"):
        from unsloth.models.remote_moe_shims import prepare_remote_moe_for_training

        prepare_remote_moe_for_training(model, verbose = False)
        block = model.model.layers[0].block_sparse_moe
        # The gate (renamed MoEGate) is shimmed through its inference-only block.
        for module in (block, block.gate):
            assert getattr(type(module).forward, "_unsloth_remote_moe_shim", False)
    return model


def _manager(
    size,
    rank,
    group = None,
):
    manager = object.__new__(cp.ContextParallelManager)
    manager.size = size
    manager.backend = "ulysses"
    manager.mesh = types.SimpleNamespace(get_group = lambda: group, get_local_rank = lambda: rank)
    return manager


def _loss(logits, shift_labels, count):
    return (
        F.cross_entropy(
            logits.to(_acc(logits)).flatten(0, 1),
            shift_labels.flatten(),
            ignore_index = -100,
            reduction = "sum",
        )
        / count
    )


def _parity_worker(rank, world, k3, control):
    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    model = _model(k3)
    ids = torch.randint(0, VOCAB, (2, 13), generator = torch.Generator().manual_seed(1))
    shift = F.pad(ids, (0, 1), value = -100)[:, 1:]
    count = shift.ne(-100).sum()
    _loss(model(input_ids = ids).logits, shift, count).backward()
    reference = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none = True)

    manager = _manager(world, rank)
    manager.attach_attention_hooks(model)
    if control:
        for module in model.modules():
            if getattr(module, "_unsloth_cp_orig_forward", None) is not None:
                module.forward = module._unsloth_cp_orig_forward
    inputs = {"input_ids": ids.clone(), "labels": ids.clone()}
    manager._prepare_inputs(inputs)
    with manager.apply(inputs):
        out = model(input_ids = inputs["input_ids"], position_ids = inputs["position_ids"])
        _loss(out.logits, inputs["shift_labels"], count).backward()
    for name, param in model.named_parameters():
        grad = torch.zeros_like(param) if param.grad is None else param.grad
        dist.all_reduce(grad)
        torch.testing.assert_close(grad, reference[name], rtol = 1e-4, atol = 1e-6, msg = name)


def _spawn(fn, world, *args):
    # Fork: a spawned child would re-import unsloth without the conftest CUDA spoof.
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs the fork start method")
    with tempfile.TemporaryDirectory() as d:
        mp.start_processes(
            _entry,
            args = (world, os.path.join(d, "init"), fn, args),
            nprocs = world,
            join = True,
            start_method = "fork",
        )


def _entry(rank, world, init_file, fn, args):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method = f"file://{init_file}", rank = rank, world_size = world)
    try:
        fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("k3", [False, True], ids = ["kimi_linear", "kimi_k3"])
def test_ulysses_kimi_gradients_match_one_rank(k3):
    _spawn(_parity_worker, 2, k3, False)


def test_shard_local_kimi_delta_is_wrong():
    # Negative control: the stock forward on a sequence shard restarts the scan at the shard seam.
    with pytest.raises(Exception, match = "AssertionError"):
        _spawn(_parity_worker, 2, False, True)


def _shard_local(fn, size, *xs):
    # Negative-control helper: run fn on each sequence shard separately, as a shard-local layer would.
    n = xs[0].shape[1] // size
    return torch.cat([fn(*(x[:, i * n : (i + 1) * n] for x in xs)) for i in range(size)], dim = 1)


# Deliberately broken copies of ulysses_kimi_delta_forward: each must fail the parity check.
_BROKEN = {
    "shard_local_scan": (
        (
            "out = _kda_scan(q, k, v, g, b.float().sigmoid())",
            "out = _shard_local(_kda_scan, size, q, k, v, g, b.float().sigmoid())",
        ),
    ),
    "shard_local_conv": tuple(
        (
            f"{x} = _kda_short_conv(module.{x}_conv1d, {x}, rows)",
            f"{x} = _shard_local(lambda t: _kda_short_conv(module.{x}_conv1d, t, rows), size, {x})",
        )
        for x in "qkv"
    ),
    "a_log_of_rank0": (
        (
            "module.A_log.reshape(-1)[rank * heads : (rank + 1) * heads]",
            "module.A_log.reshape(-1)[:heads]",
        ),
    ),
    "dt_bias_of_rank0": (
        (
            "module.dt_bias.reshape(-1).index_select(0, rows)",
            "module.dt_bias.reshape(-1)[: heads * dim]",
        ),
    ),
    "lower_bound_dropped": (('getattr(module, "gate_lower_bound", None)', "None"),),
}


def _install_broken(name):
    import inspect

    target = "ulysses_kimi_delta_forward"
    src = inspect.getsource(getattr(uh, target))
    for before, after in _BROKEN[name]:
        assert src.count(before) == 1, before
        src = src.replace(before, after)
    namespace = dict(vars(uh), _shard_local = _shard_local)
    exec(src, namespace)
    setattr(uh, target, namespace[target])


def _audit_batch(
    kind,
    seed,
    seq = 13,
):
    """Right padded rows of unequal length. "full": every valid token is a label. "seam": labels
    stop one token after the CP=2 seam (position 7), so the label at 6 crosses it and every shard
    from the second on has none."""
    ids = torch.randint(0, VOCAB, (2, seq), generator = torch.Generator().manual_seed(seed))
    mask = torch.ones_like(ids)
    mask[0, 11:] = 0
    mask[1, 9 - seed % 2 :] = 0
    labels = ids.masked_fill(mask == 0, -100)
    if kind == "seam":
        labels[:, 8:] = -100
    return ids, mask, labels


def _inject_lora(model, experts = False):
    from peft import LoraConfig, inject_adapter_in_model

    # The hub MoE: routed experts w1 / w2 / w3, the shared expert gate / up / down.
    expert_targets = ["w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"] if experts else []
    targets = expert_targets + [
        "q_proj",
        "k_proj",
        "v_proj",
        "f_b_proj",
        "b_proj",
        "g_b_proj",
        "g_proj",
        "o_proj",
        "kv_b_proj",
    ]
    model = inject_adapter_in_model(LoraConfig(r = 4, lora_alpha = 8, target_modules = targets), model)
    torch.manual_seed(2)
    for name, param in model.named_parameters():
        param.requires_grad_("lora_" in name)
        if "lora_B" in name:
            with torch.no_grad():
                param.normal_(0, 0.2)  # nonzero B: the adapters change the forward
    return model


def _audit_worker(
    rank,
    world,
    k3,
    cp_size,
    kind,
    lora,
    broken,
    masked = False,
):
    import copy

    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    dp_size = world // cp_size
    dp_rank, cp_rank = divmod(rank, cp_size)
    groups = [dist.new_group(list(range(d * cp_size, (d + 1) * cp_size))) for d in range(dp_size)]
    model = _model(k3).double()
    if lora:
        model = _inject_lora(model)
    batches = [_audit_batch(kind, seed = d) for d in range(dp_size)]
    shifts = [F.pad(labels, (0, 1), value = -100)[:, 1:] for _, _, labels in batches]
    count = sum(shift.ne(-100).sum() for shift in shifts)

    # Independent single-process fp64 reference over every DP replica's batch. masked: the delta
    # layers see the 2D padding mask as the hub model passes it (CP's decoder hook drops it).
    for (ids, mask, _), shift in zip(batches, shifts):
        _loss(
            model(input_ids = ids, attention_mask = mask if masked else None).logits, shift, count
        ).backward()
    reference = copy.deepcopy(model)  # deepcopy drops .grad
    for param, ref in zip(model.parameters(), reference.parameters()):
        ref.grad = None if param.grad is None else param.grad.clone()
    model.zero_grad(set_to_none = True)

    manager = _manager(cp_size, cp_rank, groups[dp_rank] if dp_size > 1 else None)
    manager.attach_attention_hooks(model)
    if broken is not None:
        _install_broken(broken)
    ids, mask, labels = batches[dp_rank]
    inputs = {"input_ids": ids.clone(), "attention_mask": mask.clone(), "labels": labels.clone()}
    with manager.apply(inputs):
        out = model(
            input_ids = inputs["input_ids"],
            position_ids = inputs["position_ids"],
            attention_mask = inputs["attention_mask"],
        )
        local_labels = inputs["shift_labels"].ne(-100).sum()
        _loss(out.logits, inputs["shift_labels"], count).backward()
    if kind == "seam" and cp_rank * inputs["input_ids"].shape[1] >= 7:
        assert local_labels == 0  # this shard trains on nothing of its own
    for (name, param), ref in zip(model.named_parameters(), reference.parameters()):
        # None vs zero vs value, per rank: DDP needs every rank to agree with the reference.
        assert (param.grad is None) == (ref.grad is None), name
        if param.grad is None:
            continue
        dist.all_reduce(param.grad)
        # The CP scan runs in fp32 (as fla does), the reference in fp64.
        torch.testing.assert_close(
            param.grad, ref.grad, rtol = 1e-4, atol = 1e-6, msg = lambda m: f"{name}: {m}"
        )
    if not lora:
        assert all(p.grad is not None for p in model.parameters())
    optimizers = [
        torch.optim.AdamW(
            [p for p in m.parameters() if p.requires_grad], lr = 1e-2, eps = 1e-4, weight_decay = 0.1
        )
        for m in (model, reference)
    ]
    for optimizer in optimizers:
        optimizer.step()
    for (name, param), ref in zip(model.named_parameters(), reference.parameters()):
        torch.testing.assert_close(
            param, ref, rtol = 1e-5, atol = 1e-5, msg = lambda m: f"{name} after AdamW: {m}"
        )


@pytest.mark.parametrize(
    "k3, world, cp_size, kind, lora",
    [
        (False, 2, 2, "full", False),
        (True, 2, 2, "full", False),
        (False, 2, 2, "seam", False),
        (True, 2, 2, "seam", True),
        (False, 2, 2, "full", True),
        (True, 4, 4, "full", False),
        (False, 4, 4, "seam", True),
        (True, 4, 2, "full", True),
    ],
    ids = [
        "linear-cp2",
        "k3-cp2",
        "linear-cp2-seam",
        "k3-cp2-seam-lora",
        "linear-cp2-lora",
        "k3-cp4",
        "linear-cp4-seam-lora",
        "k3-dp2xcp2-lora",
    ],
)
def test_ulysses_kimi_matches_fp64_reference(k3, world, cp_size, kind, lora):
    """Padded rows of unequal length, a shard without labels, a label across the seam, LoRA with
    nonzero B, one AdamW step: CP=2, CP=4 and DP2 x CP2 against one fp64 process."""
    _spawn(_audit_worker, world, k3, cp_size, kind, lora, None)


@pytest.mark.parametrize(
    "broken, k3",
    [
        ("shard_local_scan", False),
        ("shard_local_conv", False),
        ("a_log_of_rank0", False),
        ("dt_bias_of_rank0", True),
        ("lower_bound_dropped", True),
    ],
)
def test_broken_kimi_delta_shards_fail_the_reference(broken, k3):
    # Negative controls: a scan or conv restarted at the seam, another rank's per-head decay
    # parameters, or K3's lower-bounded gate read as the standard one.
    with pytest.raises(Exception, match = "not close"):
        _spawn(_audit_worker, 2, k3, 2, "full", False, broken)


@pytest.mark.parametrize(
    "k3, world, cp_size",
    [(False, 2, 2), (True, 4, 4)],
    ids = ["linear-cp2", "k3-cp4"],
)
def test_right_padded_kimi_under_cp_matches_a_masked_one_process(k3, world, cp_size):
    """Under CP the decoder hook (#16862) and the self_attn hook give Kimi's delta layers no padding
    mask (Kimi names them self_attn, not linear_attn). CP only accepts right padding, and the scan is
    causal, so the real tokens match a one-process run whose delta layers do mask the padding."""
    _spawn(_audit_worker, world, k3, cp_size, "full", True, None, True)


def test_the_padding_mask_changes_only_padded_positions_of_kimi_delta():
    # The control for the test above: the reference's mask is engaged (padded positions change)
    # and real tokens do not depend on the right padding after them.
    model = _model(False).double()
    ids, mask, _ = _audit_batch("full", seed = 0)
    with torch.no_grad():
        plain = model(input_ids = ids).logits
        masked = model(input_ids = ids, attention_mask = mask).logits
    real = mask.bool()
    torch.testing.assert_close(masked[real], plain[real], rtol = 1e-12, atol = 1e-12)
    assert not torch.allclose(masked[~real], plain[~real])


def test_left_padded_rows_are_refused_under_cp():
    manager = _manager(2, 0)
    ids = torch.randint(0, VOCAB, (2, 8))
    mask = torch.ones_like(ids)
    mask[0, :3] = 0  # left padding: the scan would carry pad tokens into the real ones
    with pytest.raises(ValueError, match = "right-padded"):
        manager._prepare_inputs({"input_ids": ids, "attention_mask": mask, "labels": ids.clone()})


def _moe_worker(rank, world, cp_size, precision, attn, gc):
    """Kimi-Linear with the hub MoE block (shimmed by Unsloth) under Ulysses: fp64 against one
    process over every DP replica's batch, the idle-expert hook, one AdamW step."""
    import copy

    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    dp_size = world // cp_size
    dp_rank, cp_rank = divmod(rank, cp_size)
    groups = [dist.new_group(list(range(d * cp_size, (d + 1) * cp_size))) for d in range(dp_size)]
    model = _model(False, moe = True, attn = attn, gc = gc).double()
    if precision != "full":
        model = _inject_lora(model, experts = precision == "lora_experts")
    batches = [_audit_batch("full", seed = d) for d in range(dp_size)]
    shifts = [F.pad(labels, (0, 1), value = -100)[:, 1:] for _, _, labels in batches]
    count = sum(shift.ne(-100).sum() for shift in shifts)
    for (ids, mask, _), shift in zip(batches, shifts):
        _loss(model(input_ids = ids, attention_mask = mask).logits, shift, count).backward()
    reference = copy.deepcopy(model)
    for param, ref in zip(model.parameters(), reference.parameters()):
        ref.grad = None if param.grad is None else param.grad.clone()
    model.zero_grad(set_to_none = True)
    idle = [ref.grad is None and ref.requires_grad for ref in reference.parameters()]
    # Frozen experts (attention-only LoRA) have no gradient on either side.
    assert any(idle) or precision == "lora", "no idle expert: the None vs zero check proves nothing"

    calls = {"kda": 0, "mla": 0, "experts": 0}

    def counted(key, fn):
        def wrapper(*args, **kwargs):
            calls[key] += 1
            return fn(*args, **kwargs)

        return wrapper

    uh.ulysses_kimi_delta_forward = counted("kda", uh.ulysses_kimi_delta_forward)
    uh._ulysses_hf_attention = counted("mla", uh._ulysses_hf_attention)
    cp.record_expert_tokens = counted("experts", cp.record_expert_tokens)

    manager = _manager(cp_size, cp_rank, groups[dp_rank] if dp_size > 1 else None)
    manager.attach_attention_hooks(model)
    ids, mask, labels = batches[dp_rank]
    inputs = {"input_ids": ids.clone(), "attention_mask": mask.clone(), "labels": labels.clone()}
    with manager.apply(inputs):
        out = model(
            input_ids = inputs["input_ids"],
            position_ids = inputs["position_ids"],
            attention_mask = inputs["attention_mask"],
        )
        _loss(out.logits, inputs["shift_labels"], count).backward()
    # Engaged: KDA and MLA ran their Ulysses routes, the MoE ran every expert (forward and, with
    # checkpointing, the recompute).
    assert calls["kda"] >= 2 and calls["mla"] >= 1 and calls["experts"] >= 3, calls
    trainable = [p for p in model.parameters() if p.requires_grad]
    for param in trainable:
        # Every expert ran under CP, so DDP reduces a gradient for every trainable parameter.
        assert param.grad is not None
        dist.all_reduce(param.grad)
    optimizer = torch.optim.AdamW(trainable, lr = 1e-2, eps = 1e-4, weight_decay = 0.1)
    assert cp.install_idle_expert_grad_hook(optimizer, model)
    for (name, param), ref, was_idle in zip(model.named_parameters(), reference.parameters(), idle):
        if not param.requires_grad:
            continue
        if was_idle:
            assert not param.grad.any(), name  # zero now, dropped by the hook at the step
            continue
        torch.testing.assert_close(
            param.grad, ref.grad, rtol = 1e-4, atol = 1e-6, msg = lambda m: f"{name}: {m}"
        )
    optimizer.step()
    torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad],
        lr = 1e-2,
        eps = 1e-4,
        weight_decay = 0.1,
    ).step()
    for (name, param), ref in zip(model.named_parameters(), reference.parameters()):
        assert (param.grad is None) == (ref.grad is None), name
        torch.testing.assert_close(
            param, ref, rtol = 1e-5, atol = 1e-5, msg = lambda m: f"{name} after AdamW: {m}"
        )


@pytest.mark.parametrize(
    "precision, attn, gc, world, cp_size",
    [
        ("full", "sdpa", False, 2, 2),
        ("lora", "eager", True, 2, 2),
        ("lora_experts", "sdpa", True, 2, 2),
        ("lora_experts", "eager", False, 2, 2),
        ("full", "eager", True, 4, 2),
    ],
    ids = [
        "full-sdpa-cp2",
        "lora-eager-gc-cp2",
        "lora_experts-sdpa-gc-cp2",
        "lora_experts-eager-cp2",
        "full-eager-gc-dp2xcp2",
    ],
)
def test_kimi_moe_under_ulysses_matches_one_process(precision, attn, gc, world, cp_size):
    """The parallel_capabilities evidence for Kimi-Linear MoE x context parallel (aux loss off):
    hub MoE block + gate (in-place score bias), KDA and MLA layers, right padded rows."""
    _spawn(_moe_worker, world, cp_size, precision, attn, gc)


def _pad_route_worker(rank, world, precision, attn, control):
    """128 experts, right padded rows of a length the CP size divides: caller padding routes through
    the MoE in one process, so an expert only padding picks has a zero gradient there (None when no
    token picks it). CP must give padding the same activations, or such experts differ in None vs
    zero and AdamW decays / steps a different set. Two AdamW steps, then the optimizer state."""
    import copy

    ad.HAS_FLASH_ATTENTION = False
    ad.HAS_XFORMERS = False
    if control:
        # Negative control: the Ulysses mixers leave padding as the unmasked shards compute it.
        cp.ContextParallelManager.caller_pad_lengths = lambda self, bsz, total: None
    model = _model(False, moe = 128, attn = attn).double()
    if precision != "full":
        model = _inject_lora(model, experts = True)
    reference = copy.deepcopy(model)
    manager = _manager(world, rank)
    manager.attach_attention_hooks(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    adamw = dict(lr = 1e-2, eps = 1e-4, weight_decay = 0.1)
    optimizer = torch.optim.AdamW(trainable, **adamw)
    ref_optimizer = torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad], **adamw
    )
    assert cp.install_idle_expert_grad_hook(optimizer, model)
    bad, pad_only = [], 0
    for step in range(2):
        ids, mask, labels = _audit_batch("full", seed = step, seq = 16)
        shift = F.pad(labels, (0, 1), value = -100)[:, 1:]
        count = shift.ne(-100).sum()
        optimizer.zero_grad(set_to_none = True)
        ref_optimizer.zero_grad(set_to_none = True)
        _loss(reference(input_ids = ids, attention_mask = mask).logits, shift, count).backward()
        # An expert only padding routes to: a gradient of zeros (not None) in one process.
        pad_only += sum(
            p.grad is not None and not p.grad.any()
            for n, p in reference.named_parameters()
            if ".experts." in n and "lora_B" not in n
        )
        inputs = {
            "input_ids": ids.clone(),
            "attention_mask": mask.clone(),
            "labels": labels.clone(),
        }
        with manager.apply(inputs):
            assert manager.caller_pad_lengths(2, 16) == (None if control else (11, 9 - step % 2))
            out = model(
                input_ids = inputs["input_ids"],
                position_ids = inputs["position_ids"],
                attention_mask = inputs["attention_mask"],
            )
            _loss(out.logits, inputs["shift_labels"], count).backward()
        for param in trainable:
            dist.all_reduce(param.grad)
        optimizer.step()
        ref_optimizer.step()
        for (name, param), ref in zip(model.named_parameters(), reference.parameters()):
            if (param.grad is None) != (ref.grad is None):
                bad.append((step, name, "grad None vs zero"))
            # The CP scan runs in fp32 (as fla does), the reference in fp64.
            elif param.grad is not None and not torch.allclose(param.grad, ref.grad, 1e-4, 1e-6):
                bad.append((step, name, "grad"))
            if not torch.allclose(param, ref, rtol = 1e-5, atol = 1e-5):
                bad.append((step, name, "param"))
            mine, want = optimizer.state.get(param, {}), ref_optimizer.state.get(ref, {})
            if bool(mine) != bool(want):
                bad.append((step, name, "AdamW state on one side"))
            elif mine and (
                float(mine["step"]) != float(want["step"])
                or not torch.allclose(mine["exp_avg"], want["exp_avg"], 1e-4, 1e-7)
                or not torch.allclose(mine["exp_avg_sq"], want["exp_avg_sq"], 1e-3, 1e-12)
            ):
                bad.append((step, name, "AdamW state"))
    assert pad_only, "no padding-only expert: the check proves nothing"
    assert not bad, f"CP diverges from one process: {len(bad)}, first {bad[:3]}"


@pytest.mark.parametrize(
    "precision, attn, world",
    [
        ("full", "sdpa", 2),
        ("lora_experts", "eager", 2),
        ("full", "eager", 4),
        ("lora_experts", "sdpa", 4),
    ],
    ids = ["full-sdpa-cp2", "lora_experts-eager-cp2", "full-eager-cp4", "lora_experts-sdpa-cp4"],
)
def test_kimi_caller_padding_routes_experts_as_one_process(precision, attn, world):
    _spawn(_pad_route_worker, world, precision, attn, False)


@pytest.mark.parametrize("world", [2, 4], ids = ["cp2", "cp4"])
def test_unmasked_caller_padding_misroutes_experts(world):
    # The control for the test above: without the single-process padding activations it diverges.
    with pytest.raises(Exception, match = "CP diverges from one process"):
        _spawn(_pad_route_worker, world, "full", "sdpa", True)


def test_flash_caller_padding_rows_are_zero_as_unpadded_flash():
    # transformers' flash path unpads on the 2D mask and pads the output back with zeros; a sliding
    # window under sdpa / eager is left as is (documented), real rows never change.
    out = torch.randn(2, 6, 3, 4)
    q = k = v = torch.randn(2, 3, 6, 4)

    def never(*args, **kwargs):
        raise AssertionError("flash rows need no attention call")

    got = uh._caller_padding_rows(out, None, q, k, v, (4, 6), None, "flash_attention_2", never, {})
    torch.testing.assert_close(got[0, :4], out[0, :4], rtol = 0, atol = 0)
    assert not got[0, 4:].any()
    torch.testing.assert_close(got[1], out[1], rtol = 0, atol = 0)
    assert uh._caller_padding_rows(out, None, q, k, v, (4, 6), 3, "sdpa", never, {}) is out


def test_kimi_gates():
    model = _model(False)
    kda = model.model.layers[0].self_attn
    assert uh.is_ulysses_kimi_delta(kda) and uh.is_ulysses_mixer(kda)
    assert not uh.is_ulysses_gated_delta(kda)
    assert cp._supports_context_parallel(model, "ulysses")
    assert not cp._supports_context_parallel(model, "ring")
    cp._check_stateful_mixers(model, "ulysses", 2)
    with pytest.raises(ValueError, match = "num_heads"):
        cp._check_stateful_mixers(model, "ulysses", 8)
    # MLA runs main's generic Ulysses route: its forward calls ALL_ATTENTION_FUNCTIONS.
    assert uh.is_hf_registry_attention(model.model.layers[1].self_attn)

    # An attention class next to the delta layers that bypasses the registry still blocks CP.
    def own_kernel(
        self,
        hidden_states,
        attention_mask = None,
        **kwargs,
    ):
        return hidden_states

    model.model.layers[1].self_attn.__class__ = type(
        "OtherAttention", (KimiMLAAttention,), {"forward": own_kernel}
    )
    assert not cp._supports_context_parallel(model, "ulysses")


def test_kimi_delta_refuses_packing_and_cache():
    kda = _model(False).model.layers[0].self_attn
    h = torch.zeros(1, 4, HIDDEN)
    with pytest.raises(NotImplementedError, match = "packing"):
        uh.ulysses_kimi_delta_forward(kda, h, packed_seq_lengths = torch.tensor([4]), group = None)
    with pytest.raises(RuntimeError, match = "cache"):
        uh.ulysses_kimi_delta_forward(kda, h, cache_params = object(), group = None)


def test_kimi_delta_gate_matches_fla_reference():
    f, A_log, dt = torch.randn(1, 3, 2, 4), torch.randn(2), torch.randn(8)
    x = f + dt.view(2, 4)
    expected = -A_log.exp().view(2, 1) * F.softplus(x)
    torch.testing.assert_close(uh._kda_gate(f, A_log, dt, None), expected)
    expected = -5.0 * torch.sigmoid(A_log.exp().view(2, 1) * x)
    torch.testing.assert_close(uh._kda_gate(f, A_log, dt, -5.0), expected)


def test_moe_unrouted_experts_get_zero_grads_under_context_parallelism(monkeypatch):
    from unsloth.models.remote_moe_shims import _moe_train_dispatch

    block = nn.Module()
    block.experts = nn.ModuleList(nn.Linear(4, 4) for _ in range(3))
    x = torch.randn(2, 4)
    topk_idx = torch.tensor([[0], [0]])  # experts 1 and 2 get no tokens
    topk_weight = torch.ones(2, 1)

    _moe_train_dispatch(block, x, topk_idx, topk_weight).sum().backward()
    assert block.experts[1].weight.grad is None

    block.zero_grad(set_to_none = True)
    monkeypatch.setattr(cp, "_ACTIVE_MANAGER", object())
    out = _moe_train_dispatch(block, x, topk_idx, topk_weight)
    out.sum().backward()
    for expert in block.experts[1:]:
        assert expert.weight.grad is not None and not expert.weight.grad.any()
    torch.testing.assert_close(out, block.experts[0](x))


def test_shimmed_remote_moe_is_not_blocked():
    from unsloth.models.remote_moe_shims import _moe_forward_with_training_path

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([nn.Linear(2, 2)])

        def forward(self, x):
            return x

    model = _model(False)
    model.moe = Block()
    assert "sparse MoE" in cp._hybrid_config_blocker(model)
    Block.forward = _moe_forward_with_training_path(Block.forward)
    assert cp._hybrid_config_blocker(model) is None
