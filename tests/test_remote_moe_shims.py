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
import os
import sys
import tempfile
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


def _remote_module(name = "transformers_modules.tiny_kimi.modeling_deepseek", edits = ()):
    """The port's classes, verbatim in the parts that matter, under a remote-code module name.
    ``edits`` are (old, new) source replacements for ports derived from it."""
    src = """
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
"""
    for before, after in edits:
        assert before in src, before
        src = src.replace(before, after)
    mod = types.ModuleType(name)
    # The shim predicate reads forward's source, so register the exec'd fixture in linecache.
    import linecache

    filename = f"<{name}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    exec(compile(src, filename, "exec"), mod.__dict__)
    sys.modules[name] = mod
    gate_cls = getattr(mod, "MoEGate", None) or mod.KimiMoEGate
    for cls in (gate_cls, getattr(mod, "DeepseekV3MoE", None) or mod.SarvamMLAMoE, mod.MLP):
        cls.__module__ = name
    return mod


# sarvam-105b: renamed block, `else:` still calls `moe_infer`, `num_shared_experts` key.
_SARVAM_EDITS = (
    ("class DeepseekV3MoE(nn.Module):", "class SarvamMLAMoE(nn.Module):"),
    (
        "        if not self.training:\n            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)\n",
        "        if not self.training:\n            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)\n"
        "        else:\n            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)\n",
    ),
    (
        "        if self.config.n_shared_experts is not None:",
        "        if self.shared_experts is not None:",
    ),
)


def test_detection_is_structural_and_ignores_native_classes():
    mod = _remote_module()
    block = mod.DeepseekV3MoE(mod.Cfg())
    assert is_remote_deepseek_gate(block.gate)
    assert is_remote_deepseek_moe(block)

    class MoEGate(nn.Module):  # same name, not remote code
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
    assert prepare_remote_moe_for_training(block, verbose = False) == []
    block.eval()
    with torch.no_grad():
        reference = block(x)
    block.train()
    out = block(x)
    assert torch.allclose(out, reference, atol = 1e-5, rtol = 1e-5)
    out.float().pow(2).sum().backward()
    assert block.gate.weight.grad is not None and torch.isfinite(block.gate.weight.grad).all()
    expert_grads = [
        e.down_proj.weight.grad for e in block.experts if e.down_proj.weight.grad is not None
    ]
    assert expert_grads and any(g.abs().sum() > 0 for g in expert_grads)
    assert block.shared_experts.down_proj.weight.grad is not None
    assert block.gate.training and block.training


def test_a_gate_adding_its_bias_in_place_still_backpropagates():
    """Kimi-Linear's gate (renamed KimiMoEGate) adds e_score_correction_bias in place to the sigmoid
    output that the sigmoid's backward saved; training through the shim must not trip autograd's
    version check."""
    inplace = (
        ("class MoEGate(nn.Module):", "class KimiMoEGate(nn.Module):"),
        ("self.gate = MoEGate(config)", "self.gate = KimiMoEGate(config)"),
        (
            "scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)",
            "scores_for_choice = scores.view(bsz * seq_len, -1)\n"
            "        scores_for_choice += self.e_score_correction_bias.unsqueeze(0)",
        ),
    )
    grads = []
    for name, edits in (("tiny_kimi_c", ()), ("tiny_kimi_inplace", inplace)):
        mod = _remote_module(f"transformers_modules.{name}.modeling_deepseek", edits)
        torch.manual_seed(0)
        block = mod.DeepseekV3MoE(mod.Cfg())
        assert len(prepare_remote_moe_for_training(block, verbose = False)) == 2
        block.train()
        x = torch.randn(2, 5, 16, requires_grad = True)
        block(x).float().pow(2).sum().backward()
        grads.append((x.grad, block.gate.weight.grad))
    # The bias is zero, so the in-place port routes and weighs exactly like the reference one.
    for got, expected in zip(grads[1], grads[0]):
        torch.testing.assert_close(got, expected)


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
    out = block(x)
    block.eval()
    with torch.no_grad():
        reference = block(x)
    assert torch.allclose(out, reference, atol = 1e-5, rtol = 1e-5)
    # A hook attached after the classes were shimmed is rebound too.
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
            self.language_model = LlamaForCausalLM(
                LlamaConfig(
                    hidden_size = 8,
                    num_hidden_layers = 1,
                    num_attention_heads = 2,
                    intermediate_size = 8,
                    vocab_size = 16,
                )
            )

    Outer.supports_gradient_checkpointing = False
    outer = Outer(OuterConfig())
    with pytest.raises(ValueError):
        outer.gradient_checkpointing_enable()
    assert enable_composite_gradient_checkpointing(outer, verbose = False)
    assert not enable_composite_gradient_checkpointing(outer, verbose = False)
    outer.gradient_checkpointing_enable()
    assert outer.language_model.model.gradient_checkpointing
    assert not enable_composite_gradient_checkpointing(nn.Linear(2, 2), verbose = False)


def test_a_training_capable_remote_moe_is_left_alone():
    """The standard DeepSeek-V2/V3 implementation branches on self.training and its gate
    returns an auxiliary loss; shimming it would replace a working training path."""
    import sys, types
    from unsloth.models.remote_moe_shims import is_remote_deepseek_moe

    module = types.ModuleType("transformers_modules.capable.modeling_deepseek")

    class DeepseekV3MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([nn.Linear(2, 2)])
            self.gate = nn.Linear(2, 1)

        def forward(self, hidden_states):
            if self.training:
                return hidden_states * 2
            else:
                return self.moe_infer(hidden_states)

        def moe_infer(self, x):
            return x

    DeepseekV3MoE.__module__ = module.__name__
    sys.modules[module.__name__] = module
    try:
        assert not is_remote_deepseek_moe(DeepseekV3MoE())
    finally:
        sys.modules.pop(module.__name__, None)


def test_the_gate_of_a_training_capable_block_is_not_patched():
    """A training-capable block's gate computes its auxiliary loss in train mode; the gate is
    patched only when its block is the inference-only port."""
    import linecache, sys, types
    from unsloth.models.remote_moe_shims import prepare_remote_moe_for_training

    name = "transformers_modules.capable2.modeling_deepseek"
    src = """
import torch
from torch import nn
class MoEGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.topk_method = "greedy"; self.n_group = 1; self.topk_group = 1
        self.routed_scaling_factor = 1.0; self.norm_topk_prob = False; self.top_k = 1
    def forward(self, x):
        return x
class DeepseekV3MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = MoEGate(); self.experts = nn.ModuleList([nn.Linear(2, 2)])
    def forward(self, x):
        if self.training:
            return x * 2
        else:
            return self.moe_infer(x)
    def moe_infer(self, x):
        return x
"""
    filename = f"<{name}>"
    linecache.cache[filename] = (len(src), None, src.splitlines(True), filename)
    mod = types.ModuleType(name)
    exec(compile(src, filename, "exec"), mod.__dict__)
    for cls in (mod.MoEGate, mod.DeepseekV3MoE):
        cls.__module__ = name
    sys.modules[name] = mod
    try:
        block = mod.DeepseekV3MoE()
        assert prepare_remote_moe_for_training(block, verbose = False) == []
        assert not getattr(mod.MoEGate.forward, "_unsloth_remote_moe_shim", False)
    finally:
        sys.modules.pop(name, None)


@pytest.mark.parametrize("shared", [True, False])
def test_a_renamed_port_with_a_no_grad_else_branch_is_shimmed(shared):
    """sarvam-105b trains through the same shims: its block is not called DeepseekV3MoE, its
    `else:` branch computes the output under `@torch.no_grad()` (so LoRA on the experts would
    get no gradient), and its shared expert may be None."""
    torch.manual_seed(0)
    mod = _remote_module(
        f"transformers_modules.tiny_sarvam_{int(shared)}.modeling_sarvam_moe", _SARVAM_EDITS
    )
    block = mod.SarvamMLAMoE(mod.Cfg())
    if not shared:
        block.shared_experts = None
    assert is_remote_deepseek_moe(block)
    x = torch.randn(2, 5, 16)
    block.train()
    with pytest.raises(AssertionError):
        block(x)
    patched = prepare_remote_moe_for_training(block, verbose = False)
    assert sorted(patched) == ["MoEGate", "SarvamMLAMoE"]
    block.eval()
    with torch.no_grad():
        reference = block(x)
    block.train()
    out = block(x)
    assert torch.allclose(out, reference, atol = 1e-5, rtol = 1e-5)
    out.float().pow(2).sum().backward()
    assert any(
        e.down_proj.weight.grad is not None and e.down_proj.weight.grad.abs().sum() > 0
        for e in block.experts
    )
    if shared:
        assert block.shared_experts.down_proj.weight.grad is not None


def test_an_eval_first_training_capable_remote_moe_is_left_alone():
    """`if not self.training: moe_infer(...) else: <training dispatch>` trains on its own."""
    import sys, types
    from unsloth.models.remote_moe_shims import is_remote_deepseek_moe

    module = types.ModuleType("transformers_modules.eval_first.modeling_deepseek")

    class EvalFirstMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([nn.Linear(2, 2)])
            self.gate = nn.Linear(2, 1)

        def forward(self, hidden_states):
            if not self.training:
                y = self.moe_infer(hidden_states)
            else:
                y = self.experts[0](hidden_states)
            return y

        def moe_infer(self, x):
            return x

    EvalFirstMoE.__module__ = module.__name__
    sys.modules[module.__name__] = module
    try:
        assert not is_remote_deepseek_moe(EvalFirstMoE())
    finally:
        sys.modules.pop(module.__name__, None)


@pytest.mark.parametrize(
    "training_branch, shimmed",
    [
        ('raise NotImplementedError("Training mode is not supported")', True),
        ("pass", True),
        ("y = self.experts[0](hidden_states)", False),
    ],
)
def test_a_training_branch_that_only_raises_is_not_a_dispatch(training_branch, shimmed):
    """Kimi-K3's port: `if not self.training: moe_infer(...) else: raise ...` is inference-only."""
    import sys, textwrap, types
    from unsloth.models.remote_moe_shims import is_remote_deepseek_moe

    module = types.ModuleType("transformers_modules.raise_only.modeling_kimi")
    source = textwrap.dedent(f"""
        class RaiseOnlyMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([nn.Linear(2, 2)])
                self.gate = nn.Linear(2, 1)

            def forward(self, hidden_states):
                y = hidden_states
                if not self.training:
                    y = self.moe_infer(hidden_states)
                else:
                    {training_branch}
                return y

            def moe_infer(self, x):
                return x
    """)
    path = os.path.join(tempfile.mkdtemp(dir = os.environ.get("TMPDIR")), "modeling_kimi.py")
    with open(path, "w") as f:
        f.write("import torch.nn as nn\n" + source)
    namespace = {"nn": nn, "__name__": module.__name__}
    exec(compile("import torch.nn as nn\n" + source, path, "exec"), namespace)
    RaiseOnlyMoE = namespace["RaiseOnlyMoE"]
    RaiseOnlyMoE.__module__ = module.__name__
    sys.modules[module.__name__] = module
    try:
        assert is_remote_deepseek_moe(RaiseOnlyMoE()) is shimmed
    finally:
        sys.modules.pop(module.__name__, None)


@pytest.mark.parametrize(
    "predicate, trains_in_body",
    [
        ("self.training and self.ep_size == 1", True),
        ("self.ep_size == 1 and self.training", True),
        ("not self.training and self.ep_size == 1", False),
        ("self.training or self.ep_size == 2", True),
        ("(self.training and self.ep_size == 1) or self.use_fused", True),
        ("not (self.training and self.ep_size == 1)", False),
        ("not (not self.training or self.ep_size != 1)", True),
    ],
)
def test_a_compound_training_predicate_is_recognised(predicate, trains_in_body):
    """`if self.training and self.ep_size == 1:` dispatches its own experts in training."""
    import sys, textwrap, types
    from unsloth.models.remote_moe_shims import is_remote_deepseek_moe

    module = types.ModuleType("transformers_modules.compound.modeling_deepseek")
    train, infer = "y = self.experts[0](hidden_states)", "y = self.moe_infer(hidden_states)"
    body, other = (train, infer) if trains_in_body else (infer, train)
    source = textwrap.dedent(f"""
        class CompoundMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.experts = nn.ModuleList([nn.Linear(2, 2)])
                self.gate = nn.Linear(2, 1)
                self.ep_size = 1
                self.use_fused = False

            def forward(self, hidden_states):
                if {predicate}:
                    {body}
                else:
                    {other}
                return y

            def moe_infer(self, x):
                return x
    """)
    path = os.path.join(tempfile.mkdtemp(dir = os.environ.get("TMPDIR")), "modeling_compound.py")
    with open(path, "w") as f:
        f.write("import torch.nn as nn\n" + source)
    module.__file__ = path
    namespace = {"nn": nn, "__name__": module.__name__}
    exec(compile("import torch.nn as nn\n" + source, path, "exec"), namespace)
    CompoundMoE = namespace["CompoundMoE"]
    CompoundMoE.__module__ = module.__name__
    sys.modules[module.__name__] = module
    try:
        assert not is_remote_deepseek_moe(CompoundMoE())
    finally:
        sys.modules.pop(module.__name__, None)


_KIMI_SRC = """
import torch
import torch.nn.functional as F
from torch import nn

class Cfg:
    hidden_size = 16; routed_expert_hidden_size = 8; moe_intermediate_size = 12
    num_experts = 6; num_experts_per_token = 2; num_shared_experts = 1; routed_scaling_factor = 2.0

class KimiRMSNorm(nn.Module):
    def __init__(self, n, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(n)); self.variance_epsilon = eps
    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (self.weight * (x.float() * torch.rsqrt(v + self.variance_epsilon))).to(x.dtype)

class KimiMLP(nn.Module):
    def __init__(self, hidden, inter):
        super().__init__()
        self.w1 = nn.Linear(hidden, inter, bias=False); self.w3 = nn.Linear(hidden, inter, bias=False)
        self.w2 = nn.Linear(inter, hidden, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class KimiMoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_token; self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.randn(config.num_experts, config.hidden_size) * 0.3)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(config.num_experts))
    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        logits = F.linear(hidden_states.view(-1, h).float(), self.weight.float())
        scores = logits.sigmoid()
        assert not self.training
        _, topk_idx = torch.topk(scores + self.e_score_correction_bias, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_idx, topk_weight * self.routed_scaling_factor

class KimiSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config; self.ep_size = 1; self.ep_rank = 0; self.experts_per_rank = config.num_experts
        self.experts = nn.ModuleList([KimiMLP(config.routed_expert_hidden_size, config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.gate = KimiMoEGate(config)
        self.shared_experts = KimiMLP(config.hidden_size, config.moe_intermediate_size)
        self.routed_expert_down_proj = nn.Linear(config.hidden_size, config.routed_expert_hidden_size, bias=False)
        self.routed_expert_up_proj = nn.Linear(config.routed_expert_hidden_size, config.hidden_size, bias=False)
        self.routed_expert_norm = KimiRMSNorm(config.routed_expert_hidden_size)
    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.routed_expert_down_proj(hidden_states)
        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight)
        else:
            raise NotImplementedError("Training mode is not supported in KimiSparseMoeBlock")
        y = self.routed_expert_norm(y)
        y = self.routed_expert_up_proj(y)
        y = y.view(*orig_shape)
        if self.config.num_shared_experts is not None:
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
        return new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype).mul_(topk_weight.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype)
"""


def _kimi_module(name):
    import linecache

    filename = f"<{name}>"
    linecache.cache[filename] = (len(_KIMI_SRC), None, _KIMI_SRC.splitlines(True), filename)
    mod = types.ModuleType(name)
    exec(compile(_KIMI_SRC, filename, "exec"), mod.__dict__)
    for cls in (mod.KimiRMSNorm, mod.KimiMLP, mod.KimiMoEGate, mod.KimiSparseMoeBlock):
        cls.__module__ = name
    sys.modules[name] = mod
    return mod


def test_a_latent_moe_trains_through_its_own_projections():
    torch.manual_seed(0)
    mod = _kimi_module("transformers_modules.tiny_kimi_k3.modeling_kimi_linear")
    block = mod.KimiSparseMoeBlock(mod.Cfg())
    x = torch.randn(2, 5, 16)
    block.train()
    with pytest.raises((AssertionError, NotImplementedError)):
        block(x)
    # The renamed gate is shimmed through its inference-only block (in-place-safe backward).
    assert sorted(prepare_remote_moe_for_training(block, verbose = False)) == [
        "KimiMoEGate",
        "KimiSparseMoeBlock",
    ]
    block.eval()
    with torch.no_grad():
        reference = block(x)
    block.train()
    out = block(x)
    assert torch.allclose(out, reference, atol = 1e-5, rtol = 1e-5)
    out.float().pow(2).sum().backward()
    for p in (
        block.gate.weight,
        block.routed_expert_down_proj.weight,
        block.routed_expert_up_proj.weight,
        block.routed_expert_norm.weight,
        block.shared_experts.w2.weight,
    ):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert any(
        e.w2.weight.grad is not None and e.w2.weight.grad.abs().sum() > 0 for e in block.experts
    )
    assert block.training and block.gate.training
    assert "moe_infer" not in vars(block)
    block.eval()
    with torch.no_grad():
        assert torch.allclose(block(x), reference)


_KIMI_INPLACE = (
    ("class MoEGate(nn.Module):", "class KimiMoEGate(nn.Module):"),
    ("self.gate = MoEGate(config)", "self.gate = KimiMoEGate(config)"),
    (
        "scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)",
        "scores_for_choice = scores.view(bsz * seq_len, -1)\n"
        "        scores_for_choice += self.e_score_correction_bias.unsqueeze(0)",
    ),
)
# The same gate written out of place. The in-place add goes through a view, so Kimi's `scores`
# (later gathered into the routing weights) carries the bias too: keep that.
_KIMI_OUT_OF_PLACE = _KIMI_INPLACE[:2] + (
    (
        "scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)",
        "scores = scores + self.e_score_correction_bias.unsqueeze(0)\n"
        "        scores_for_choice = scores.view(bsz * seq_len, -1)",
    ),
    ("torch.float32", "torch.float64"),
)


def _kimi_gate_block(name, edits, scoring, dtype):
    if scoring == "softmax":
        edits = edits + (("scores = logits.sigmoid()", "scores = logits.softmax(dim=-1)"),)
    mod = _remote_module(f"transformers_modules.{name}.modeling_deepseek", edits)
    torch.manual_seed(0)
    block = mod.DeepseekV3MoE(mod.Cfg())
    with torch.no_grad():
        block.gate.e_score_correction_bias.normal_(0, 0.3)  # nonzero: Kimi's real bias
    assert sorted(prepare_remote_moe_for_training(block, verbose = False)) == [
        "DeepseekV3MoE",
        "KimiMoEGate",
    ]
    return block.to(dtype).train()


def _checkpointed(block, x, mode):
    from torch.utils.checkpoint import checkpoint

    if mode == "none":
        return block(x)
    if mode == "reentrant":
        return checkpoint(block, x, use_reentrant = True)
    if mode == "non_reentrant":
        return checkpoint(block, x, use_reentrant = False)
    # How vision.py runs non-reentrant checkpointing under DDP: inside the zoo offload hooks.
    from unsloth_zoo.gradient_checkpointing import non_reentrant_offload_hooks

    with non_reentrant_offload_hooks():
        return checkpoint(block, x, use_reentrant = False)


@pytest.mark.parametrize("scoring", ["sigmoid", "softmax"])
@pytest.mark.parametrize("mode", ["none", "reentrant", "non_reentrant", "non_reentrant_offload"])
def test_an_in_place_kimi_gate_matches_an_out_of_place_oracle(mode, scoring):
    """Kimi-Linear adds its (nonzero) score bias in place to the sigmoid / softmax output that op's
    backward saved. Without a copy, plain and reentrant backward raise autograd's version error and
    non-reentrant checkpointing silently differentiates the biased scores."""
    if mode == "non_reentrant_offload":
        pytest.importorskip("unsloth_zoo.gradient_checkpointing").non_reentrant_offload_hooks
    x = torch.randn(3, 7, 16, dtype = torch.float64, generator = torch.Generator().manual_seed(1))
    oracle = _kimi_gate_block(f"kimi_oop_{scoring}", _KIMI_OUT_OF_PLACE, scoring, torch.float64)
    x_oracle = x.clone().requires_grad_()
    expected = oracle(x_oracle)
    expected.pow(2).sum().backward()

    block = _kimi_gate_block(
        f"kimi_inplace_{mode}_{scoring}", _KIMI_INPLACE, scoring, torch.float32
    )
    x_block = x.float().requires_grad_()
    out = _checkpointed(block, x_block, mode)
    out.pow(2).sum().backward()
    torch.testing.assert_close(out.double(), expected, rtol = 1e-5, atol = 1e-6)
    torch.testing.assert_close(x_block.grad.double(), x_oracle.grad, rtol = 1e-4, atol = 1e-6)
    for (name, got), want in zip(block.named_parameters(), oracle.parameters()):
        assert (got.grad is None) == (want.grad is None), name
        if got.grad is not None:
            torch.testing.assert_close(
                got.grad.double(), want.grad, rtol = 1e-4, atol = 1e-6, msg = lambda m: f"{name}: {m}"
            )


def test_the_gate_leaves_its_saved_tensors_to_checkpointing():
    """Inside non-reentrant checkpointing (Unsloth's DDP / context parallel default) every saved
    tensor belongs to the checkpoint, which drops it until backward. saved_tensors_hooks pushed
    inside the gate would win over the checkpoint's and keep the gate's saved tensors (its fp32
    input copy, scores, weights) alive through the whole forward."""
    import gc
    import weakref

    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils.checkpoint import checkpoint

    class Created(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.refs = []

        def __torch_dispatch__(
            self,
            func,
            types,
            args = (),
            kwargs = None,
        ):
            out = func(*args, **(kwargs or {}))
            for t in out if isinstance(out, (tuple, list)) else (out,):
                if isinstance(t, torch.Tensor):
                    self.refs.append(weakref.ref(t))
            return out

    block = _kimi_gate_block("kimi_resident", _KIMI_INPLACE, "sigmoid", torch.float32)
    x = torch.randn(2, 64, 16, requires_grad = True)
    with Created() as created:
        out = checkpoint(block, x, use_reentrant = False)
    gc.collect()
    keep = {t.untyped_storage().data_ptr() for t in (out, x, *block.parameters())}
    resident = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
        for t in (r() for r in created.refs)
        if t is not None and t.untyped_storage().data_ptr() not in keep
    }
    assert sum(resident.values()) == 0, resident
    out.pow(2).sum().backward()
    assert x.grad is not None and block.gate.weight.grad is not None


def test_a_non_remote_gate_is_not_patched():
    mod = _remote_module("transformers_modules.kimi_local_gate.modeling_deepseek", _KIMI_INPLACE)
    mod.KimiMoEGate.__module__ = "my_project.modeling"  # a local class an inference-only port uses
    original = mod.KimiMoEGate.forward
    block = mod.DeepseekV3MoE(mod.Cfg())
    assert prepare_remote_moe_for_training(block, verbose = False) == ["DeepseekV3MoE"]
    assert mod.KimiMoEGate.forward is original
    assert prepare_remote_moe_for_training(block, verbose = False) == []


def _moe_ddp_entry(rank, world, init_file, fn, args):
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method = f"file://{init_file}",
        rank = rank,
        world_size = world,
        timeout = datetime.timedelta(seconds = 60),
    )
    try:
        fn(rank, world, *args)
    finally:
        dist.destroy_process_group()


def _moe_spawn(fn, world, *args):
    import torch.multiprocessing as mp
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("needs the fork start method")
    with tempfile.TemporaryDirectory() as d:
        mp.start_processes(
            _moe_ddp_entry,
            args = (world, os.path.join(d, "init"), fn, args),
            nprocs = world,
            join = True,
            start_method = "fork",
        )


def _routed_tokens(experts, seed):
    # The gate below scores expert e by feature e, so these tokens pick exactly `experts`.
    x = 0.1 * torch.randn(1, 6, 16, generator = torch.Generator().manual_seed(seed))
    for boost, expert in zip((5.0, 4.0), experts):
        x[..., expert] += boost
    return x


def _complementary_block(lora):
    mod = _remote_module("transformers_modules.kimi_ddp.modeling_deepseek", _KIMI_INPLACE)
    torch.manual_seed(0)
    block = mod.DeepseekV3MoE(mod.Cfg())
    with torch.no_grad():
        block.gate.weight.copy_(torch.eye(8, 16))
    prepare_remote_moe_for_training(block, verbose = False)
    if lora:
        from peft import LoraConfig, inject_adapter_in_model

        config = LoraConfig(r = 2, target_modules = r".*experts\.\d+\.(gate_proj|up_proj|down_proj)")
        block = inject_adapter_in_model(config, block)
        for name, param in block.named_parameters():
            param.requires_grad_("lora_" in name)
            if "lora_B" in name:
                with torch.no_grad():
                    param.normal_(0, 0.1)
    return block.train()


# Per step, the experts each rank routes to: disjoint across ranks and changing every step, so
# with the shard-local routing of context parallelism each rank leaves experts its peer trains.
_ROUTES = (((0, 1), (2, 3)), ((2, 3), (4, 5)), ((6, 7), (0, 1)))


def _moe_ddp_worker(rank, world, cp_active, lora):
    import copy

    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    from unsloth import context_parallel as cp

    block = _complementary_block(lora)
    reference = copy.deepcopy(block)
    model = DistributedDataParallel(block, find_unused_parameters = False)
    params = [p for p in block.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = 0.1)
    ref_optimizer = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr = 0.1)
    cp._ACTIVE_MANAGER = object() if cp_active else None
    try:
        for step, routes in enumerate(_ROUTES):
            optimizer.zero_grad(set_to_none = True)
            model(_routed_tokens(routes[rank], step * 10 + rank)).pow(2).sum().backward()
            # DDP averages over ranks: the reference sees both ranks' tokens at half weight.
            ref_optimizer.zero_grad(set_to_none = True)
            for peer in range(world):
                loss = reference(_routed_tokens(routes[peer], step * 10 + peer)).pow(2).sum()
                (loss / world).backward()
            used = {e for pair in routes for e in pair}
            for (name, got), want in zip(block.named_parameters(), reference.parameters()):
                if not got.requires_grad:
                    continue
                expert = int(name.split(".")[1]) if name.startswith("experts.") else None
                assert got.grad is not None, name  # a zero gradient, never None, for DDP
                if want.grad is None:
                    assert expert is not None and expert not in used, name
                    assert not got.grad.any(), name
                else:
                    torch.testing.assert_close(got.grad, want.grad, msg = lambda m: f"{name}: {m}")
            optimizer.step()
            ref_optimizer.step()
        for (name, got), want in zip(block.named_parameters(), reference.parameters()):
            torch.testing.assert_close(got, want, msg = lambda m: f"{name} after 3 steps: {m}")
    finally:
        cp._ACTIVE_MANAGER = None


@pytest.mark.parametrize("lora", [False, True], ids = ["full", "lora_experts"])
def test_context_parallel_ranks_routing_disjoint_experts_train_under_ddp(lora):
    """Under context parallelism each rank routes its own sequence shard, so an expert idle on one
    rank trains on its peer. DDP with find_unused_parameters = False needs a gradient from every
    rank: idle experts run on no tokens for a zero gradient, over several steps of shifting routes."""
    _moe_spawn(_moe_ddp_worker, 2, True, lora)


def test_without_context_parallelism_idle_experts_stall_ddp():
    # Negative control: the CP-off dispatch leaves idle experts' gradients None, so DDP without
    # find_unused_parameters never reduces that bucket (local gradients) and refuses the next step.
    with pytest.raises(Exception, match = "not close|finished reduction|did not receive grad"):
        _moe_spawn(_moe_ddp_worker, 2, False, False)


# Per optimizer step, two gradient accumulation micro-batches, the experts each rank routes to.
# Step 0: 6 and 7 idle everywhere; 4 and 5 used in the second micro-batch only. Step 1: 2 to 5
# idle everywhere after training at step 0 (AdamW momentum and weight decay would move them).
# Step 2: 4 to 7 idle everywhere.
_GA_ROUTES = (
    (((0, 1), (2, 3)), ((0, 1), (4, 5))),
    (((6, 7), (0, 1)), ((0, 1), (6, 7))),
    (((2, 3), (2, 3)), ((0, 1), (2, 3))),
)


def _moe_adamw_worker(rank, world, lora, hook):
    import copy

    from torch.nn.parallel import DistributedDataParallel

    from unsloth import context_parallel as cp

    block = _complementary_block(lora)
    reference = copy.deepcopy(block)
    model = DistributedDataParallel(block, find_unused_parameters = False)
    adamw = dict(lr = 1e-2, weight_decay = 0.1)
    optimizer = torch.optim.AdamW([p for p in block.parameters() if p.requires_grad], **adamw)
    ref_optimizer = torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad], **adamw
    )
    if hook:
        assert cp.install_idle_expert_grad_hook(optimizer, model)
        assert cp.install_idle_expert_grad_hook(optimizer, model)  # once per optimizer
        assert len(optimizer._optimizer_step_pre_hooks) == 1
    cp._ACTIVE_MANAGER = object()
    try:
        for step, micro_routes in enumerate(_GA_ROUTES):
            optimizer.zero_grad(set_to_none = True)
            ref_optimizer.zero_grad(set_to_none = True)
            for micro, routes in enumerate(micro_routes):
                seed = step * 100 + micro * 10
                loss = model(_routed_tokens(routes[rank], seed + rank)).pow(2).sum()
                (loss / len(micro_routes)).backward()
                cp._ACTIVE_MANAGER = None  # the single-process reference
                for peer in range(world):
                    loss = reference(_routed_tokens(routes[peer], seed + peer)).pow(2).sum()
                    (loss / (world * len(micro_routes))).backward()
                cp._ACTIVE_MANAGER = object()
            used = {e for routes in micro_routes for pair in routes for e in pair}
            before = {n: p.detach().clone() for n, p in block.named_parameters()}
            optimizer.step()
            ref_optimizer.step()
            for (name, got), want in zip(block.named_parameters(), reference.parameters()):
                if not got.requires_grad:
                    continue
                expert = int(name.split(".")[1]) if name.startswith("experts.") else None
                if hook and expert is not None:
                    assert (got.grad is None) == (expert not in used), (step, name)
                if expert is not None and expert not in used:
                    assert torch.equal(got, before[name]), f"idle {name} moved at step {step}"
                torch.testing.assert_close(got, want, msg = lambda m: f"{name} step {step}: {m}")
    finally:
        cp._ACTIVE_MANAGER = None


@pytest.mark.parametrize("lora", [False, True], ids = ["full", "lora_experts"])
def test_context_parallel_idle_experts_follow_single_process_adamw(lora):
    """An expert no rank routes to runs on no tokens under CP, so DDP hands it a zero gradient.
    The optimizer step pre-hook must turn that back into None so AdamW (weight decay, moments,
    step count) treats it like the single-process run, over gradient accumulation micro-batches
    where an expert used in only one micro-batch keeps its gradient."""
    _moe_spawn(_moe_adamw_worker, 2, lora, True)


def test_without_the_hook_idle_experts_move_under_adamw():
    # Negative control: the zero gradients alone decay idle experts and advance their moments.
    with pytest.raises(Exception, match = "moved at step|not close"):
        _moe_spawn(_moe_adamw_worker, 2, False, False)


def test_idle_expert_hook_is_inert_without_every_expert_blocks_and_when_all_are_routed():
    from unsloth import context_parallel as cp

    optimizer = torch.optim.AdamW(nn.Linear(2, 2).parameters())
    assert not cp.install_idle_expert_grad_hook(optimizer, nn.Linear(2, 2))
    assert not optimizer._optimizer_step_pre_hooks
    block = _complementary_block(False)
    cp._ACTIVE_MANAGER = object()
    try:
        x = torch.cat([_routed_tokens((e, (e + 1) % 8), e) for e in range(0, 8, 2)], dim = 1)
        block(x).pow(2).sum().backward()
        with torch.no_grad():  # a graph-free forward does not count
            block(_routed_tokens((0, 1), 0))
    finally:
        cp._ACTIVE_MANAGER = None
    assert block._unsloth_cp_expert_tokens.gt(0).all()
    grads = {n: p.grad.clone() for n, p in block.named_parameters() if p.grad is not None}
    assert cp.drop_idle_expert_grads([block]) == 0
    assert "_unsloth_cp_expert_tokens" not in block.__dict__
    for n, p in block.named_parameters():
        if n in grads:
            assert torch.equal(p.grad, grads[n]), n
    # No counts recorded (CP off): every expert counts as idle, but CP never installs the hook.
    block(_routed_tokens((0, 1), 0)).pow(2).sum().backward()
    assert "_unsloth_cp_expert_tokens" not in block.__dict__


def test_an_expert_parallel_block_is_not_counted_as_running_every_expert():
    """ep_size > 1 keeps the remote forward (its own expert all-to-all), not the zero-token
    dispatch, so the context parallel MoE refusal must still apply to it."""
    from unsloth import context_parallel as cp

    mod = _remote_module("transformers_modules.kimi_ep.modeling_deepseek", _KIMI_INPLACE)
    block = mod.DeepseekV3MoE(mod.Cfg())
    prepare_remote_moe_for_training(block, verbose = False)
    assert cp._runs_every_expert(block)
    block.ep_size = 2
    assert not cp._runs_every_expert(block)
    block.train()
    with pytest.raises(Exception):  # the port has no training path of its own
        block(torch.randn(1, 3, 16))
