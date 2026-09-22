# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training forwards for remote DeepSeek-V3-style MoE code (Kimi-K2.7) and the gradient
checkpointing flag on a remote composition. Runs offline on CPU with a synthetic copy of the
port's `MoEGate` / `DeepseekV3MoE` idioms defined in a `transformers_modules` namespace."""

import math
import sys
import types

import pytest
import torch
from torch import nn
from torch.nn import functional as F

# Import unsloth first to set UNSLOTH_IS_PRESENT env var.
import unsloth  # noqa: F401
from unsloth.models.remote_moe_shims import (
    is_remote_deepseek_gate,
    is_remote_deepseek_moe,
    prepare_remote_moe_for_training,
)
from unsloth.models.loader_utils import enable_composite_gradient_checkpointing


def _remote_module(name = "transformers_modules.tiny_kimi.modeling_deepseek"):
    """The port's classes, verbatim in the parts that matter, under a remote-code module name."""
    src = '''
import math, torch
import torch.nn.functional as F
from torch import nn

class Cfg:
    def __init__(self):
        self.hidden_size = 16; self.n_routed_experts = 8; self.num_experts_per_tok = 2
        self.moe_intermediate_size = 8; self.n_shared_experts = 1; self.routed_scaling_factor = 2.5
        self.scoring_func = "sigmoid"; self.seq_aux = True; self.topk_method = "noaux_tc"
        self.n_group = 1; self.topk_group = 1; self.norm_topk_prob = True; self.hidden_act = "silu"

class MLP(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config; self.top_k = config.num_experts_per_tok; self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor; self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method; self.n_group = config.n_group; self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob; self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.e_score_correction_bias = nn.Parameter(torch.zeros((self.n_routed_experts)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        scores = logits.sigmoid()
        assert not self.training
        scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores); group_mask.scatter_(1, group_idx, 1)
        score_mask = group_mask.unsqueeze(-1).expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group).reshape(bsz * seq_len, -1)
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight * self.routed_scaling_factor

class DeepseekV3MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config; self.num_experts_per_tok = config.num_experts_per_tok
        self.ep_size = 1; self.experts_per_rank = config.n_routed_experts; self.ep_rank = 0
        self.experts = nn.ModuleList([MLP(config, config.moe_intermediate_size) for _ in range(config.n_routed_experts)])
        self.gate = MoEGate(config)
        self.shared_experts = MLP(config, config.moe_intermediate_size * config.n_shared_experts)
    def forward(self, hidden_states):
        identity = hidden_states; orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y
    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts))); cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0); idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        outputs = []; start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert.cpu().numpy()):
            end_idx = start_idx + num_tokens
            if num_tokens == 0: continue
            outputs.append(self.experts[i](sorted_tokens[start_idx:end_idx])); start_idx = end_idx
        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs); new_x[idxs] = outs
        return (new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype).mul_(topk_weight.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype))
'''
    mod = types.ModuleType(name)
    exec(src, mod.__dict__)
    sys.modules[name] = mod
    for cls in (mod.MoEGate, mod.DeepseekV3MoE, mod.MLP):
        cls.__module__ = name
    return mod


def test_detection_is_structural_and_ignores_native_classes():
    mod = _remote_module()
    block = mod.DeepseekV3MoE(mod.Cfg())
    assert is_remote_deepseek_gate(block.gate)
    assert is_remote_deepseek_moe(block)

    class MoEGate(nn.Module):   # same name, not remote code
        pass

    assert not is_remote_deepseek_gate(MoEGate())
    assert not is_remote_deepseek_moe(nn.Linear(2, 2))


def test_training_forward_matches_eval_and_backpropagates():
    torch.manual_seed(0)
    mod = _remote_module("transformers_modules.tiny_kimi_b.modeling_deepseek")
    block = mod.DeepseekV3MoE(mod.Cfg())
    x = torch.randn(2, 5, 16)
    block.train()
    with pytest.raises(AssertionError):
        block(x)
    patched = prepare_remote_moe_for_training(block, verbose = False)
    assert sorted(patched) == ["DeepseekV3MoE", "MoEGate"]
    assert prepare_remote_moe_for_training(block, verbose = False) == []   # idempotent
    block.eval()
    with torch.no_grad():
        reference = block(x)
    block.train()
    out = block(x)
    assert torch.allclose(out, reference, atol = 1e-5, rtol = 1e-5)
    out.float().pow(2).sum().backward()
    assert block.gate.weight.grad is not None and torch.isfinite(block.gate.weight.grad).all()
    expert_grads = [e.down_proj.weight.grad for e in block.experts if e.down_proj.weight.grad is not None]
    assert expert_grads and any(g.abs().sum() > 0 for g in expert_grads)
    assert block.shared_experts.down_proj.weight.grad is not None
    assert block.gate.training and block.training   # the gate flag is restored after the call


def test_shims_reach_a_module_behind_an_accelerate_hook():
    """A device_map load wraps forward with an accelerate hook that keeps the original as
    `_old_forward`; the class-level shim alone is never called from `module(...)`. Kimi-K2.7-Code
    on four GPUs still hit the gate's training assert this way."""
    from accelerate.hooks import ModelHook, add_hook_to_module

    torch.manual_seed(0)
    mod = _remote_module("transformers_modules.tiny_kimi_c.modeling_deepseek")
    block = mod.DeepseekV3MoE(mod.Cfg())
    for sub in (block, block.gate):
        add_hook_to_module(sub, ModelHook())
    assert "_old_forward" in vars(block.gate)
    block.train()
    x = torch.randn(2, 5, 16)
    with pytest.raises(AssertionError):
        block(x)
    prepare_remote_moe_for_training(block, verbose = False)
    out = block(x)          # goes through the hook, which must now reach the shim
    block.eval()
    with torch.no_grad():
        reference = block(x)
    assert torch.allclose(out, reference, atol = 1e-5, rtol = 1e-5)
    # A hook attached after the classes were shimmed (a second model of the same remote code) is rebound too.
    block2 = mod.DeepseekV3MoE(mod.Cfg())
    add_hook_to_module(block2.gate, ModelHook())
    block2.train()
    assert prepare_remote_moe_for_training(block2, verbose = False) == []
    block2(x)


def test_composite_gradient_checkpointing_flag():
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedModel, PretrainedConfig

    class OuterConfig(PretrainedConfig):
        model_type = "outer_test"

    class Outer(PreTrainedModel):
        config_class = OuterConfig

        def __init__(self, config):
            super().__init__(config)
            self.language_model = LlamaForCausalLM(LlamaConfig(hidden_size = 8, num_hidden_layers = 1, num_attention_heads = 2, intermediate_size = 8, vocab_size = 16))

    Outer.supports_gradient_checkpointing = False
    outer = Outer(OuterConfig())
    with pytest.raises(ValueError):
        outer.gradient_checkpointing_enable()
    assert enable_composite_gradient_checkpointing(outer, verbose = False)
    assert not enable_composite_gradient_checkpointing(outer, verbose = False)
    outer.gradient_checkpointing_enable()
    assert outer.language_model.model.gradient_checkpointing
    assert not enable_composite_gradient_checkpointing(nn.Linear(2, 2), verbose = False)
