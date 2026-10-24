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
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + 1e-6)
        return (self.weight * x * torch.sigmoid(g.float())).to(g.dtype)


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
        f = self.f_b_proj(self.f_a_proj(hidden_states)).view(shape).float() + self.dt_bias.view(
            HEADS, HEAD_DIM
        )
        a = self.A_log.float().exp().view(HEADS, 1)
        if self.gate_lower_bound is not None:
            g = self.gate_lower_bound * torch.sigmoid(a * f)
        else:
            g = -a * F.softplus(f)
        beta = self.b_proj(hidden_states).float().sigmoid()
        q = F.normalize(q.float(), dim = -1) * HEAD_DIM**-0.5
        k = F.normalize(k.float(), dim = -1)
        state = q.new_zeros(bsz, HEADS, HEAD_DIM, HEAD_DIM)
        outs = []
        for t in range(seq):
            state = state * g[:, t].exp().unsqueeze(-1)
            pred = (k[:, t].unsqueeze(-1) * state).sum(-2)
            state = state + k[:, t].unsqueeze(-1) * (
                beta[:, t, :, None] * (v[:, t].float() - pred)
            ).unsqueeze(-2)
            outs.append((q[:, t].unsqueeze(-1) * state).sum(-2))
        o = torch.stack(outs, dim = 1)
        if hasattr(self, "g_proj"):
            gate = self.g_proj(hidden_states)
        else:
            gate = self.g_b_proj(self.g_a_proj(hidden_states))
        o = self.o_norm(o, gate.view(shape))
        return self.o_proj(o.reshape(bsz, seq, -1))


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
        attention = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        out, _ = attention(self, q, key, value, attention_mask, dropout = 0.0, scaling = self.scaling)
        out = out.reshape(bsz, seq, -1)
        if self.use_output_gate:
            out = out * self.g_proj(hidden_states).sigmoid()
        return self.o_proj(out)


class _Layer(nn.Module):
    def __init__(self, config, linear, k3):
        super().__init__()
        self.self_attn = KimiDeltaAttention(k3) if linear else KimiMLAAttention(config, k3)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN, 2 * HIDDEN), nn.SiLU(), nn.Linear(2 * HIDDEN, HIDDEN)
        )
        self.norm1, self.norm2 = nn.LayerNorm(HIDDEN), nn.LayerNorm(HIDDEN)

    def forward(
        self,
        h,
        attention_mask = None,
    ):
        h = h + self.self_attn(hidden_states = self.norm1(h), attention_mask = attention_mask)
        return h + self.mlp(self.norm2(h))


class _KimiForCausalLM(nn.Module):
    def __init__(self, k3):
        super().__init__()
        self.config = types.SimpleNamespace(
            _attn_implementation = "sdpa",
            num_attention_heads = HEADS,
            num_key_value_heads = HEADS,
            output_router_logits = False,
            vision_config = None,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            _Layer(self.config, linear, k3) for linear in (True, False, True)
        )
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias = False)

    def forward(
        self,
        input_ids,
        position_ids = None,
        use_cache = False,
        **kwargs,
    ):
        h = self.embed(input_ids)
        for layer in self.model.layers:
            h = layer(h)
        return types.SimpleNamespace(logits = self.lm_head(h))


def _model(k3):
    torch.manual_seed(0)
    return _KimiForCausalLM(k3).train()


def _manager(size, rank):
    manager = object.__new__(cp.ContextParallelManager)
    manager.size = size
    manager.backend = "ulysses"
    manager.mesh = types.SimpleNamespace(get_group = lambda: None, get_local_rank = lambda: rank)
    return manager


def _loss(logits, shift_labels, count):
    return (
        F.cross_entropy(
            logits.float().flatten(0, 1), shift_labels.flatten(), ignore_index = -100, reduction = "sum"
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
    # An unknown attention class next to the delta layers still blocks CP.
    model.model.layers[1].self_attn.__class__ = type("OtherAttention", (KimiMLAAttention,), {})
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


def test_mla_value_heads_are_padded_for_ulysses(monkeypatch):
    seen = {}

    def fake_ulysses(run, config, context, q, k, v, group):
        seen["v"] = v.shape[-1]
        return torch.zeros(q.shape[0], q.shape[2], q.shape[1], q.shape[3])

    monkeypatch.setattr(uh, "ulysses_attention", fake_ulysses)
    module = types.SimpleNamespace(training = False)
    q = torch.zeros(1, HEADS, 3, 12)
    out, _ = uh._ulysses_hf_attention(module, q, q, torch.zeros(1, HEADS, 3, 8), None, group = None)
    assert seen["v"] == 12 and out.shape[-1] == 8
