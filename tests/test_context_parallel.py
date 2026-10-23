# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Context parallel for Qwen3.5 on CPU (gloo): 2 ranks must match 1 process."""

import os
import socket
import tempfile

try:
    import pytest
except ImportError:
    import types

    pytest = types.ModuleType("pytest")
    pytest.importorskip = lambda name: __import__(name)

    class _Mark:
        @staticmethod
        def parametrize(*args, **kwargs):
            return lambda fn: fn

    pytest.mark = _Mark()
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("transformers.models.qwen3_5")


def _load_parallel():
    # Load unsloth/distributed on its own: spawned ranks cannot run unsloth's
    # accelerator check on a CPU-only runner, and the package does not need it.
    import importlib.util
    import pathlib
    import sys

    name = "_unsloth_distributed_under_test"
    if name not in sys.modules:
        root = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "distributed"
        spec = importlib.util.spec_from_file_location(
            name, root / "__init__.py", submodule_search_locations = [str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return importlib.import_module(f"{name}.parallel")


P = _load_parallel()


def _tiny_model():
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5

    if hasattr(modeling_qwen3_5.torch_chunk_gated_delta_rule, "__wrapped__"):
        modeling_qwen3_5.torch_chunk_gated_delta_rule = (
            modeling_qwen3_5.torch_chunk_gated_delta_rule.__wrapped__
        )
    if hasattr(modeling_qwen3_5.causal_conv1d_fn, "__wrapped__"):
        modeling_qwen3_5.causal_conv1d_fn = modeling_qwen3_5.causal_conv1d_fn.__wrapped__

    config = Qwen3_5TextConfig(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        layer_types = ["linear_attention", "full_attention"],
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        linear_num_key_heads = 2,
        linear_num_value_heads = 4,
        linear_key_head_dim = 16,
        linear_value_head_dim = 16,
    )
    torch.manual_seed(3407)
    model = Qwen3_5ForCausalLM(config).float()
    model.config.use_cache = False
    for mod in model.modules():
        fn = getattr(mod, "chunk_gated_delta_rule", None)
        if fn is not None and hasattr(fn, "__wrapped__"):
            mod.chunk_gated_delta_rule = fn.__wrapped__
    return model


def _batch(seq):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 128, (2, seq), generator = g)
    labels = ids.clone()
    labels[:, :3] = -100
    return ids, labels


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _worker(rank, world, port, seq, out):
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, init_method = f"tcp://127.0.0.1:{port}"
    )
    try:
        model = _tiny_model()
        P.apply_cp(model, dist.group.WORLD)
        ids, labels = _batch(seq)
        loss = P.sft_loss(model, ids, labels, dist.group.WORLD)
        loss.backward()
        P.sync_replicated_grads(model, dist.group.WORLD)
        if rank == 0:
            torch.save({"loss": loss.detach(), "grads": _grads(model)}, out)
    finally:
        dist.destroy_process_group()


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("seq", [32, 33])
def test_cp_matches_single_process(seq):
    model = _tiny_model()
    ids, labels = _batch(seq)
    ref = P.sft_loss(model, ids, labels, None)
    ref.backward()
    ref_grads = _grads(model)

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "cp.pt")
        mp.spawn(_worker, args = (2, _free_port(), seq, out), nprocs = 2, join = True)
        got = torch.load(out)

    torch.testing.assert_close(got["loss"], ref.detach(), rtol = 1e-5, atol = 1e-5)
    assert got["grads"].keys() == ref_grads.keys()
    for name, grad in ref_grads.items():
        torch.testing.assert_close(got["grads"][name], grad, rtol = 1e-4, atol = 1e-6, msg = name)


def test_packed_and_padded_masks():
    seq = 8
    pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    keep = P._blocked_key_mask(None, pos, 1, seq, "cpu")
    assert not bool(keep[0, 0, 4, 0]) and not bool(keep[0, 0, 7, 3])
    assert bool(keep[0, 0, 5, 4]) and not bool(keep[0, 0, 5, 6])

    pad = torch.ones(1, seq, dtype = torch.long)
    pad[0, 6:] = 0
    keep_pad = P._blocked_key_mask(pad, None, 1, seq, "cpu")
    assert not bool(keep_pad[0, 0, 3, 6]) and bool(keep_pad[0, 0, 3, 1])

    assert P._blocked_key_mask(None, torch.arange(seq).view(1, -1), 1, seq, "cpu") is None

    torch.manual_seed(0)
    q = torch.randn(1, seq, 2, 8, requires_grad = True)
    k = torch.randn(1, seq, 1, 8, requires_grad = True)
    v = torch.randn(1, seq, 1, 8, requires_grad = True)
    out = P.sdpa_flash_fn(q, k, v, None, is_causal = True, position_ids = pos)
    out[:, 4:].sum().backward()
    assert v.grad[:, :4].abs().max().item() == 0
    assert v.grad[:, 4:].abs().max().item() > 0


def _tiny_deepseek_v2():
    from transformers import DeepseekV2Config, DeepseekV2ForCausalLM

    config = DeepseekV2Config(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        moe_intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        kv_lora_rank = 16,
        q_lora_rank = 24,
        qk_rope_head_dim = 8,
        v_head_dim = 16,
        qk_nope_head_dim = 16,
        n_routed_experts = 4,
        n_shared_experts = 1,
        num_experts_per_tok = 2,
        first_k_dense_replace = 1,
        rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 4.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 64,
        },
    )
    torch.manual_seed(3407)
    model = DeepseekV2ForCausalLM(config).float()
    model.config.use_cache = False
    return model


def _tiny_deepseek_v32():
    """DeepSeek-V3.2: MLA + DSA indexer. index_topk < seq so the sparse selection actually bites."""
    from transformers import DeepseekV32Config, DeepseekV32ForCausalLM

    config = DeepseekV32Config(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        moe_intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        kv_lora_rank = 16,
        q_lora_rank = 24,
        qk_rope_head_dim = 8,
        v_head_dim = 16,
        qk_nope_head_dim = 16,
        n_routed_experts = 4,
        n_shared_experts = 1,
        num_experts_per_tok = 2,
        n_group = 1,
        topk_group = 1,
        first_k_dense_replace = 1,
        index_topk = 4,
        index_head_dim = 16,
        index_n_heads = 2,
        rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 4.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
            "original_max_position_embeddings": 64,
        },
    )
    torch.manual_seed(3407)
    model = DeepseekV32ForCausalLM(config).float()
    model.config.use_cache = False
    return model


def _tiny_deepseek_v4():
    from transformers.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM

    config = DeepseekV4Config(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        moe_intermediate_size = 32,
        num_hidden_layers = 3,
        num_attention_heads = 4,
        num_key_value_heads = 1,
        head_dim = 32,
        rope_head_dim = 16,
        q_lora_rank = 32,
        o_lora_rank = 32,
        o_groups = 2,
        num_local_experts = 4,
        num_experts_per_tok = 2,
        sliding_window = 8,
        index_n_heads = 2,
        index_head_dim = 16,
        index_topk = 2,
        layer_types = [
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        mlp_layer_types = ["hash_moe", "moe", "moe"],
        compress_rates = {
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 8,
        },
    )
    torch.manual_seed(3407)
    model = DeepseekV4ForCausalLM(config).float()
    with torch.no_grad():
        g = torch.Generator().manual_seed(3407)
        for mod in model.modules():
            if hasattr(mod, "sinks"):
                mod.sinks.copy_(torch.randn(mod.sinks.shape, generator = g) * 0.1)
            if hasattr(mod, "position_bias"):
                mod.position_bias.copy_(torch.randn(mod.position_bias.shape, generator = g) * 0.05)
            if hasattr(mod, "tid2eid"):
                tok_ids = torch.arange(config.vocab_size)
                mod.tid2eid[:, 0] = tok_ids % config.num_local_experts
                mod.tid2eid[:, 1] = (tok_ids + 1) % config.num_local_experts
    model.config.use_cache = False
    return model


def _tiny_minimax():
    from transformers.models.minimax import MiniMaxConfig, MiniMaxForCausalLM

    config = MiniMaxConfig(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        num_local_experts = 4,
        num_experts_per_tok = 2,
        block_size = 8,
        layer_types = ["linear_attention", "full_attention"],
        full_attn_alpha_factor = 3.5,
        full_attn_beta_factor = 1.0,
        linear_attn_alpha_factor = 3.5,
        linear_attn_beta_factor = 1.0,
        mlp_alpha_factor = 3.5,
        mlp_beta_factor = 1.0,
    )
    torch.manual_seed(3407)
    model = MiniMaxForCausalLM(config).float()
    model.config.use_cache = False
    return model


def _tiny_minimax_m2():
    from transformers.models.minimax_m2 import MiniMaxM2Config, MiniMaxM2ForCausalLM

    config = MiniMaxM2Config(
        vocab_size = 128,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 16,
        num_local_experts = 4,
        num_experts_per_tok = 2,
        bos_token_id = 1,
        eos_token_id = 2,
    )
    torch.manual_seed(3407)
    model = MiniMaxM2ForCausalLM(config).float()
    model.config.use_cache = False
    return model


def _tiny_gemma4():
    from transformers.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig

    config = Gemma4TextConfig(
        vocab_size = 128,
        vocab_size_per_layer_input = 128,
        hidden_size = 64,
        hidden_size_per_layer_input = 16,
        intermediate_size = 128,
        num_hidden_layers = 4,
        num_kv_shared_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 1,
        head_dim = 16,
        global_head_dim = 32,
        sliding_window = 8,
        final_logit_softcapping = 30.0,
        use_double_wide_mlp = True,
        layer_types = [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    torch.manual_seed(3407)
    model = Gemma4ForCausalLM(config).float()
    model.config.use_cache = False
    return model


def _tiny_gemma4_moe_k_eq_v():
    """Gemma-4 26B-A4B / 31B shape: k_eq_v global layers (fewer KV heads than cp), MoE, no PLE / KV sharing."""
    from transformers.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig

    config = Gemma4TextConfig(
        vocab_size = 128,
        vocab_size_per_layer_input = 128,
        hidden_size = 64,
        hidden_size_per_layer_input = 0,
        intermediate_size = 128,
        num_hidden_layers = 4,
        num_kv_shared_layers = 0,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        num_global_key_value_heads = 1,
        head_dim = 16,
        global_head_dim = 32,
        attention_k_eq_v = True,
        sliding_window = 8,
        final_logit_softcapping = 30.0,
        enable_moe_block = True,
        num_experts = 4,
        top_k_experts = 2,
        moe_intermediate_size = 32,
        layer_types = [
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    torch.manual_seed(3407)
    model = Gemma4ForCausalLM(config).float()
    model.config.use_cache = False
    return model


def _tiny_kimi_linear(k3 = False):
    from types import SimpleNamespace
    import torch.nn as nn
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    import importlib

    patch_kimi_model = importlib.import_module(f"{P.__package__}.kda_cp").patch_kimi_model

    class _ShortConv(nn.Conv1d):
        def __init__(self, hidden_size, kernel_size, activation = "silu"):
            super().__init__(
                hidden_size,
                hidden_size,
                kernel_size,
                groups = hidden_size,
                bias = False,
                padding = kernel_size - 1,
            )
            self.activation = activation

    class _RMSNormGated(nn.Module):
        def __init__(self, hidden_size, eps = 1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x, g):
            dt = x.dtype
            xf = x.float()
            xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim = True) + self.eps)
            return (self.weight * xf * torch.sigmoid(g.float())).to(dt)

    class KimiRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps = 1e-5):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            dt = hidden_states.dtype
            hidden_states = hidden_states.float()
            var = hidden_states.pow(2).mean(-1, keepdim = True)
            return (self.weight * hidden_states * torch.rsqrt(var + self.variance_epsilon)).to(dt)

    class KimiDeltaAttention(nn.Module):
        def __init__(self, cfg, layer_idx):
            super().__init__()
            self.config = cfg
            self.layer_idx = layer_idx
            self.hidden_size = cfg.hidden_size
            self.conv_size = cfg.linear_attn_config["short_conv_kernel_size"]
            self.head_dim = cfg.linear_attn_config["head_dim"]
            self.num_heads = cfg.linear_attn_config["num_heads"]
            self.head_k_dim = self.head_dim
            self.num_k_heads = self.num_heads
            proj = self.head_dim * self.num_heads
            self.q_proj = nn.Linear(self.hidden_size, proj, bias = False)
            self.k_proj = nn.Linear(self.hidden_size, proj, bias = False)
            self.v_proj = nn.Linear(self.hidden_size, proj, bias = False)
            self.q_conv1d = _ShortConv(proj, self.conv_size)
            self.k_conv1d = _ShortConv(proj, self.conv_size)
            self.v_conv1d = _ShortConv(proj, self.conv_size)
            self.A_log = nn.Parameter(torch.zeros(1, 1, self.num_heads, 1))
            self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias = False)
            self.f_b_proj = nn.Linear(self.head_dim, proj, bias = False)
            self.dt_bias = nn.Parameter(torch.zeros(proj))
            self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias = False)
            self.use_full_rank_gate = cfg.linear_attn_config.get("use_full_rank_gate", False)
            self.gate_lower_bound = cfg.linear_attn_config.get("gate_lower_bound", None)
            if self.use_full_rank_gate:
                self.g_proj = nn.Linear(self.hidden_size, proj, bias = False)
            else:
                self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias = False)
                self.g_b_proj = nn.Linear(self.head_dim, proj, bias = False)
            if k3:
                # non-trivial values so per-rank slicing of A_log / dt_bias is actually exercised
                with torch.no_grad():
                    self.A_log.copy_(torch.randn(1, 1, self.num_heads, 1) * 0.5)
                    self.dt_bias.copy_(torch.randn(proj) * 0.5)
            self.o_norm = _RMSNormGated(self.head_dim, eps = cfg.rms_norm_eps)
            self.o_proj = nn.Linear(proj, self.hidden_size, bias = False)

    class KimiMLAAttention(nn.Module):
        def __init__(self, cfg, layer_idx):
            super().__init__()
            self.config = cfg
            self.layer_idx = layer_idx
            self.hidden_size = cfg.hidden_size
            self.num_heads = cfg.num_attention_heads
            self.num_key_value_heads = cfg.num_key_value_heads
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
            self.qk_nope_head_dim = cfg.qk_nope_head_dim
            self.qk_rope_head_dim = cfg.qk_rope_head_dim
            self.v_head_dim = cfg.v_head_dim
            self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
            self.kv_lora_rank = cfg.kv_lora_rank
            self.scaling = self.q_head_dim ** -0.5
            self.attention_dropout = 0.0
            self.is_causal = True
            self.q_lora_rank = getattr(cfg, "q_lora_rank", None)
            if self.q_lora_rank is not None:
                self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias = False)
                self.q_a_layernorm = KimiRMSNorm(self.q_lora_rank)
                self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.q_head_dim, bias = False)
            else:
                self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias = False)
            self.use_output_gate = getattr(cfg, "mla_use_output_gate", False)
            if self.use_output_gate:
                self.g_proj = nn.Linear(self.hidden_size, self.num_heads * self.v_head_dim, bias = False)
            self.kv_a_proj_with_mqa = nn.Linear(
                self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias = False
            )
            self.kv_a_layernorm = KimiRMSNorm(self.kv_lora_rank)
            self.kv_b_proj = nn.Linear(
                self.kv_lora_rank,
                self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
                bias = False,
            )
            self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias = False)

        def forward(self, hidden_states, attention_mask = None, **kwargs):
            bsz, seq_len, _ = hidden_states.shape
            q_states = (
                self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
                if self.q_lora_rank is not None
                else self.q_proj(hidden_states)
            )
            q = q_states.view(bsz, seq_len, self.num_heads, self.q_head_dim).transpose(1, 2)
            q_pass, q_rot = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim = -1)
            ckv = self.kv_a_proj_with_mqa(hidden_states)
            k_pass, k_rot = torch.split(ckv, [self.kv_lora_rank, self.qk_rope_head_dim], dim = -1)
            k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(
                bsz, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            ).transpose(1, 2)
            k_pass, v = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim = -1)
            k_rot = k_rot.view(bsz, 1, seq_len, self.qk_rope_head_dim).expand_as(q_rot)
            q_states = torch.cat((q_pass, q_rot), dim = -1)
            k_states = torch.cat((k_pass, k_rot), dim = -1)
            if self.config._attn_implementation == "flash_attention_2" and self.q_head_dim != self.v_head_dim:
                v = torch.nn.functional.pad(v, [0, self.q_head_dim - self.v_head_dim])
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
            out, _ = attn_fn(
                self, q_states, k_states, v, attention_mask, dropout = 0.0, scaling = self.scaling, **kwargs
            )
            if self.config._attn_implementation == "flash_attention_2" and self.q_head_dim != self.v_head_dim:
                out = out[:, :, :, : self.v_head_dim]
            out = out.reshape(bsz, seq_len, -1).contiguous()
            if self.use_output_gate:
                out = out * self.g_proj(hidden_states).sigmoid()
            return self.o_proj(out)

    class KimiBlockSparseMLP(nn.Module):
        def __init__(self, hidden_dim, ffn_dim):
            super().__init__()
            self.w1 = nn.Linear(hidden_dim, ffn_dim, bias = False)
            self.w2 = nn.Linear(ffn_dim, hidden_dim, bias = False)
            self.w3 = nn.Linear(hidden_dim, ffn_dim, bias = False)

        def forward(self, x):
            return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))

    class KimiMoEGate(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.top_k = cfg.num_experts_per_token
            self.num_experts = cfg.num_experts
            self.routed_scaling_factor = cfg.routed_scaling_factor
            self.moe_router_activation_func = "sigmoid"
            self.num_expert_group = 1
            self.topk_group = 1
            self.moe_renormalize = True
            self.weight = nn.Parameter(torch.empty(self.num_experts, cfg.hidden_size))
            self.e_score_correction_bias = nn.Parameter(torch.zeros(self.num_experts))
            nn.init.kaiming_uniform_(self.weight, a = 5 ** 0.5)

    class KimiSparseMoeBlock(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self.ep_rank = 0
            self.experts_per_rank = cfg.num_experts
            self.moe_hidden = getattr(cfg, "routed_expert_hidden_size", None) or cfg.hidden_size
            self.experts = nn.ModuleList(
                [KimiBlockSparseMLP(self.moe_hidden, cfg.moe_intermediate_size) for _ in range(cfg.num_experts)]
            )
            self.gate = KimiMoEGate(cfg)
            self.shared_experts = KimiBlockSparseMLP(cfg.hidden_size, cfg.moe_intermediate_size)
            if self.moe_hidden != cfg.hidden_size:
                self.routed_expert_down_proj = nn.Linear(cfg.hidden_size, self.moe_hidden, bias = False)
                self.routed_expert_up_proj = nn.Linear(self.moe_hidden, cfg.hidden_size, bias = False)

    class KimiSparseMoeBlockK3(KimiSparseMoeBlock):
        # K3 ships its own trainable MoE; patch_kimi_model must leave it alone.
        def moe_train(self, x, topk_ids, topk_weight):
            full = torch.zeros(x.shape[0], len(self.experts), dtype = topk_weight.dtype).scatter(
                1, topk_ids, topk_weight
            )
            y = torch.zeros_like(x)
            for e, expert in enumerate(self.experts):
                y = y + expert(x) * full[:, e : e + 1].to(x.dtype)
            return y

        def forward(self, hidden_states):
            shape = hidden_states.shape
            flat = hidden_states.reshape(-1, shape[-1])
            scores = torch.sigmoid(flat.float() @ self.gate.weight.float().T)
            _, topk_ids = torch.topk(scores + self.gate.e_score_correction_bias, self.gate.top_k, dim = -1)
            topk_weight = scores.gather(1, topk_ids)
            topk_weight = topk_weight / (topk_weight.sum(-1, keepdim = True) + 1e-20)
            x = self.routed_expert_down_proj(flat) if hasattr(self, "routed_expert_down_proj") else flat
            y = self.moe_train(x, topk_ids, topk_weight * self.gate.routed_scaling_factor)
            if hasattr(self, "routed_expert_up_proj"):
                y = self.routed_expert_up_proj(y)
            return y.view(shape) + self.shared_experts(hidden_states)

    class KimiDecoderLayer(nn.Module):
        def __init__(self, cfg, idx):
            super().__init__()
            self.is_linear_attn = idx == 0
            self.self_attn = KimiDeltaAttention(cfg, idx) if self.is_linear_attn else KimiMLAAttention(cfg, idx)
            self.block_sparse_moe = (KimiSparseMoeBlockK3 if k3 else KimiSparseMoeBlock)(cfg)
            self.input_layernorm = KimiRMSNorm(cfg.hidden_size)
            self.post_attention_layernorm = KimiRMSNorm(cfg.hidden_size)

        def forward(self, h, position_ids = None):
            h = h + self.self_attn(self.input_layernorm(h), position_ids = position_ids)
            h = h + self.block_sparse_moe(self.post_attention_layernorm(h))
            return h

    class KimiLinearModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
            self.layers = nn.ModuleList([KimiDecoderLayer(cfg, i) for i in range(2)])
            self.norm = KimiRMSNorm(cfg.hidden_size)

        def forward(self, input_ids = None, position_ids = None, use_cache = False):
            h = self.embed_tokens(input_ids)
            for layer in self.layers:
                h = layer(h, position_ids = position_ids)
            return SimpleNamespace(last_hidden_state = self.norm(h))

    class KimiLinearForCausalLM(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self.model = KimiLinearModel(cfg)
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias = False)

    cfg = SimpleNamespace(
        model_type = "kimi_linear",
        vocab_size = 128,
        hidden_size = 64,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        qk_nope_head_dim = 16,
        qk_rope_head_dim = 8,
        v_head_dim = 16,
        kv_lora_rank = 32,
        num_experts = 4,
        num_experts_per_token = 2,
        moe_intermediate_size = 32,
        num_shared_experts = 1,
        routed_scaling_factor = 1.0,
        rms_norm_eps = 1e-5,
        linear_attn_config = {
            "short_conv_kernel_size": 4,
            "head_dim": 16,
            "num_heads": 4,
            **({"use_full_rank_gate": True, "gate_lower_bound": -5.0} if k3 else {}),
        },
        **(
            dict(q_lora_rank = 24, mla_use_output_gate = True, routed_expert_hidden_size = 48)
            if k3
            else {}
        ),
        _attn_implementation = "sdpa",
        use_cache = False,
    )
    torch.manual_seed(3407)
    model = KimiLinearForCausalLM(cfg).float()
    patch_kimi_model(model)
    return model


def _tiny_kimi_k3():
    return _tiny_kimi_linear(k3 = True)


def _generic_worker(rank, world, port, seq, out, builder):
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, init_method = f"tcp://127.0.0.1:{port}"
    )
    try:
        model = builder()
        P.apply_cp(model, dist.group.WORLD)
        ids, labels = _batch(seq)
        loss = P.sft_loss(model, ids, labels, dist.group.WORLD)
        loss.backward()
        P.sync_replicated_grads(model, dist.group.WORLD)
        if rank == 0:
            torch.save({"loss": loss.detach(), "grads": _grads(model)}, out)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "builder",
    [
        _tiny_deepseek_v2,
        _tiny_deepseek_v32,
        _tiny_deepseek_v4,
        _tiny_minimax,
        _tiny_minimax_m2,
        _tiny_gemma4,
        _tiny_gemma4_moe_k_eq_v,
        _tiny_kimi_linear,
        _tiny_kimi_k3,
    ],
)
def test_deepseek_cp_matches_single_process(builder):
    seq = 16
    model = builder()
    ids, labels = _batch(seq)
    ref = P.sft_loss(model, ids, labels, None)
    ref.backward()
    ref_grads = _grads(model)

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "cp.pt")
        mp.spawn(_generic_worker, args = (2, _free_port(), seq, out, builder), nprocs = 2, join = True)
        got = torch.load(out)

    torch.testing.assert_close(got["loss"], ref.detach(), rtol = 1e-4, atol = 1e-5)
    assert got["grads"].keys() == ref_grads.keys()
    for name, grad in ref_grads.items():
        torch.testing.assert_close(got["grads"][name], grad, rtol = 1e-4, atol = 1e-5, msg = name)


if __name__ == "__main__":
    test_packed_and_padded_masks()
    print("test_packed_and_padded_masks: PASS")
    for s in (32, 33):
        test_cp_matches_single_process(s)
        print(f"test_cp_matches_single_process(seq={s}): PASS")
    for b in (
        _tiny_deepseek_v2,
        _tiny_deepseek_v32,
        _tiny_deepseek_v4,
        _tiny_minimax,
        _tiny_minimax_m2,
        _tiny_gemma4,
        _tiny_gemma4_moe_k_eq_v,
        _tiny_kimi_linear,
        _tiny_kimi_k3,
    ):
        test_deepseek_cp_matches_single_process(b)
        print(f"test_deepseek_cp_matches_single_process({b.__name__}): PASS")
