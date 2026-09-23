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

"""A compressed-tensors MXFP4 checkpoint whose routed experts are remote-code w1 / w2 / w3
Linears (Kimi-K3) keeps them packed: the per-expert bytes are stacked into one unsloth_zoo
`Mxfp4StackedExperts` per MoE layer at load, never decompressed or re-quantized to NF4, and the
remote MoE block dispatches to it in train and eval. UNSLOTH_MXFP4_KEEP_PACKED=0 restores the
NF4 path. Offline tests cover the key scan, the swap, the stacking op, the LoRA target choice and
the dispatch; the GPU test loads a tiny remote-code checkpoint through transformers."""

import json
import os
import sys
import types

import pytest
import torch
from torch import nn

# Import unsloth first to set UNSLOTH_IS_PRESENT env var.
import unsloth  # noqa: F401
from unsloth.models.compressed_tensors_bnb import (
    _StackPackedExperts,
    _transformers_supports_weight_converters,
    arm_compressed_tensors_bnb_loading,
    keep_mxfp4_experts_packed,
    packed_expert_prefixes,
    swap_in_packed_mxfp4_experts,
)
from unsloth.models.remote_moe_shims import (
    is_remote_deepseek_moe,
    packed_expert_target_parameters,
    prepare_remote_moe_for_training,
)

zoo_stacked = pytest.importorskip("unsloth_zoo.mxfp4_stacked_experts")
Mxfp4StackedExperts = zoo_stacked.Mxfp4StackedExperts

try:
    import compressed_tensors  # noqa: F401
    HAS_CT = True
except Exception:
    HAS_CT = False

# The quantizer hook that adds converters (HfQuantizer.update_weight_conversions) is 5.8+.
HAS_CONVERTERS = _transformers_supports_weight_converters()

H, I, E, K = 64, 64, 4, 2

MODELING = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput


class TinyMoeConfig(PretrainedConfig):
    model_type = "tiny_mxfp4_moe"

    def __init__(self, vocab_size=128, hidden_size=64, moe_intermediate_size=64, num_experts=4,
                 num_experts_per_tok=2, num_hidden_layers=2, hidden_act="situ", **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        super().__init__(**kwargs)


class SituAndMul(nn.Module):
    def forward(self, x):
        d = x.shape[-1] // 2
        gate, up = x[..., :d].float(), x[..., d:].float()
        return (torch.tanh(gate) * torch.sigmoid(gate) * up).to(x.dtype)


class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.w2 = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.act_fn = SituAndMul()

    def forward(self, x):
        return self.w2(self.act_fn(torch.cat([self.w1(x), self.w3(x)], dim=-1)))


class Gate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.weight = nn.Parameter(torch.randn(config.num_experts, config.hidden_size) * 0.1)

    def forward(self, hidden_states):
        h = hidden_states.view(-1, hidden_states.shape[-1])
        scores = F.linear(h.float(), self.weight.float()).softmax(dim=-1)
        assert not self.training
        weight, idx = scores.topk(self.top_k, dim=-1)
        return idx, weight


class SparseMoe(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.gate = Gate(config)

    def forward(self, hidden_states):
        shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, shape[-1])
        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight)
        else:
            raise NotImplementedError("inference only")
        return y.view(*shape)

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0).cpu().numpy()
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        outputs, start = [], 0
        for i, n in enumerate(tokens_per_expert):
            if n == 0:
                continue
            outputs.append(self.experts[i](sorted_tokens[start:start + n]))
            start += n
        outs = torch.cat(outputs, dim=0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        return (new_x.view(*topk_ids.shape, -1).type(topk_weight.dtype)
                .mul_(topk_weight.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype))


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mlp = SparseMoe(config)

    def forward(self, x):
        return x + self.mlp(self.proj(x))


class TinyMoeForCausalLM(PreTrainedModel):
    config_class = TinyMoeConfig
    base_model_prefix = "model"
    _no_split_modules = ["Layer"]

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        pass

    def forward(self, input_ids, labels=None, **kwargs):
        h = self.embed(input_ids)
        for layer in self.layers:
            h = layer(h)
        logits = self.lm_head(h).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return CausalLMOutput(loss=loss, logits=logits)
"""


def _mxfp4_plan():
    return {
        "quant_method": "compressed-tensors",
        "format": "mxfp4-pack-quantized",
        "quantization_status": "compressed",
        "config_groups": {
            "group_0": {
                "format": "mxfp4-pack-quantized",
                "targets": ["Linear"],
                "input_activations": None,
                "output_activations": None,
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "group",
                    "group_size": 32,
                    "symmetric": True,
                    "dynamic": False,
                    "scale_dtype": "torch.uint8",
                    "actorder": None,
                    "observer": "minmax",
                },
            }
        },
        "ignore": ["re:.*\\.proj$", "re:.*lm_head.*"],
    }


def _remote_module(name, source = MODELING):
    import linecache

    mod = types.ModuleType(name)
    filename = f"<{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), mod.__dict__)
    sys.modules[name] = mod
    for cls in (
        mod.SparseMoe,
        mod.Gate,
        mod.Expert,
        mod.Layer,
        mod.TinyMoeForCausalLM,
        mod.SituAndMul,
    ):
        cls.__module__ = name
    return mod


def _keys(
    layers = 2,
    experts = E,
    drop = None,
    extra = (),
):
    keys = []
    for layer in range(layers):
        for e in range(experts):
            for w in ("w1", "w2", "w3"):
                for kind in ("weight_packed", "weight_scale"):
                    keys.append(f"model.layers.{layer}.mlp.experts.{e}.{w}.{kind}")
        keys.append(f"model.layers.{layer}.proj.weight")
    if drop:
        keys.remove(drop)
    return keys + list(extra)


def test_key_scan_needs_every_expert_projection_and_nothing_else():
    assert packed_expert_prefixes(_keys()) == {"model.layers.0.mlp": E, "model.layers.1.mlp": E}
    # A partly packed layer disables the whole scan: its remaining keys would otherwise be
    # claimed by the stacking converters and dropped as unexpected.
    assert packed_expert_prefixes(_keys(drop = "model.layers.1.mlp.experts.2.w3.weight_scale")) == {}
    extra = ("model.layers.0.mlp.experts.0.w1.weight_shape",)
    assert packed_expert_prefixes(_keys(extra = extra)) == {}
    assert packed_expert_prefixes(["model.layers.0.proj.weight"]) == {}


def test_keep_packed_is_mxfp4_only_and_switchable(monkeypatch):
    monkeypatch.delenv("UNSLOTH_MXFP4_KEEP_PACKED", raising = False)
    assert keep_mxfp4_experts_packed(_mxfp4_plan())
    int4 = dict(_mxfp4_plan(), format = "pack-quantized")
    int4["config_groups"] = {
        "group_0": dict(int4["config_groups"]["group_0"], format = "pack-quantized")
    }
    assert not keep_mxfp4_experts_packed(int4)
    monkeypatch.setenv("UNSLOTH_MXFP4_KEEP_PACKED", "0")
    assert not keep_mxfp4_experts_packed(_mxfp4_plan())


def _tiny_model(name, layers = 2, source = MODELING):
    mod = _remote_module(name, source)
    config = mod.TinyMoeConfig(num_hidden_layers = layers)
    with torch.device("meta"):
        model = mod.TinyMoeForCausalLM(config)
    return mod, model


def test_swap_replaces_every_packed_layer_on_meta():
    mod, model = _tiny_model("transformers_modules.k3c_swap_a.modeling_tinymoe")
    swapped = swap_in_packed_mxfp4_experts(model, _keys(), torch.bfloat16)
    assert swapped == ["layers.0.mlp", "layers.1.mlp"]
    for layer in model.layers:
        experts = layer.mlp.experts
        assert isinstance(experts, Mxfp4StackedExperts)
        assert experts.fused_gate_up_act and len(experts) == E
        assert experts.gate_up_blocks.shape == (E, 2 * I, H // 32, 16)
        assert experts.down_scales.shape == (E, H, I // 32)
        assert experts.gate_up_blocks.dtype == torch.uint8 and experts.gate_up_blocks.is_meta
        assert isinstance(layer.proj, nn.Linear)  # non-expert Linears are left to bitsandbytes


def test_swap_is_all_or_nothing():
    _, model = _tiny_model("transformers_modules.k3c_swap_b.modeling_tinymoe")
    # Layer 1's experts are not all packed: layer 0 must not be swapped either, so no
    # stacking converter can claim layer 1's keys.
    keys = _keys(drop = "model.layers.1.mlp.experts.0.w2.weight_packed")
    assert swap_in_packed_mxfp4_experts(model, keys, torch.bfloat16) == []
    # One packed layer that no module matches.
    keys = _keys(layers = 3)
    assert swap_in_packed_mxfp4_experts(model, keys, torch.bfloat16) == []
    assert all(isinstance(layer.mlp.experts, nn.ModuleList) for layer in model.layers)
    # Experts with a bias are not the plain w1 / w2 / w3 layout.
    _, model = _tiny_model("transformers_modules.k3c_swap_c.modeling_tinymoe")
    for layer in model.layers:
        for expert in layer.mlp.experts:
            expert.w2 = nn.Linear(I, H, bias = True, device = "meta")
    assert swap_in_packed_mxfp4_experts(model, _keys(), torch.bfloat16) == []


def test_stacking_op_keeps_w1_then_w3_and_expert_order():
    g = torch.Generator().manual_seed(0)
    per = {
        w: [torch.randint(0, 256, (I, H // 2), dtype = torch.uint8, generator = g) for _ in range(E)]
        for w in ("w1", "w3")
    }
    sources = ["experts.*.w1.weight_packed$", "experts.*.w3.weight_packed$"]
    got = _StackPackedExperts(False).convert(
        {sources[1]: per["w3"], sources[0]: per["w1"]},
        source_patterns = sources,
        target_patterns = ["experts.gate_up_blocks"],
    )["experts.gate_up_blocks"]
    assert got.shape == (E, 2 * I, H // 32, 16)
    for e in range(E):
        assert torch.equal(got[e, :I].reshape(I, H // 2), per["w1"][e])
        assert torch.equal(got[e, I:].reshape(I, H // 2), per["w3"][e])
    scales = [torch.randint(0, 256, (H, I // 32), dtype = torch.uint8, generator = g) for _ in range(E)]
    got = _StackPackedExperts(True).convert(
        {"experts.*.w2.weight_scale$": scales},
        source_patterns = ["experts.*.w2.weight_scale$"],
        target_patterns = ["experts.down_scales"],
    )["experts.down_scales"]
    assert torch.equal(got, torch.stack(scales))


def test_expert_lora_stays_opt_in():
    _, model = _tiny_model("transformers_modules.k3c_lora.modeling_tinymoe")
    swap_in_packed_mxfp4_experts(model, _keys(), torch.bfloat16)
    auto = ["experts.gate_up_proj", "experts.down_proj"]
    assert packed_expert_target_parameters(model, auto, ["q_proj", "down_proj"]) is None
    assert packed_expert_target_parameters(model, auto, ["w1"]) == ["experts.gate_up_proj"]
    assert packed_expert_target_parameters(model, None, ["w2", "w3"]) == auto
    assert packed_expert_target_parameters(model, ["mlp.other"], None) == ["mlp.other"]
    # No packed experts: untouched.
    _, plain = _tiny_model("transformers_modules.k3c_lora_b.modeling_tinymoe")
    assert packed_expert_target_parameters(plain, auto, None) is auto


def test_packed_experts_are_found_under_peft_wrappers():
    """Expert LoRA wraps `block.experts` in PEFT ParamWrappers (one per parameter); the block
    must still dispatch to the packed stacks, or it falls back to the port's `len(experts)`."""
    from unsloth.models.remote_moe_shims import _is_packed_experts

    _, model = _tiny_model("transformers_modules.k3c_wrapped.modeling_tinymoe")
    swap_in_packed_mxfp4_experts(model, _keys(), torch.bfloat16)

    class Wrapper(nn.Module):
        def __init__(self, base_layer):
            super().__init__()
            self.base_layer = base_layer

    block = model.layers[0].mlp
    block.experts = Wrapper(Wrapper(block.experts))
    assert _is_packed_experts(block.experts)
    assert is_remote_deepseek_moe(block)


def _materialize_packed(model, seed = 0):
    """Fill the meta stacks with random MXFP4 bytes (scales near 1) and finalize."""
    g = torch.Generator().manual_seed(seed)
    for layer in model.layers:
        experts = layer.mlp.experts
        for name, param in list(experts.named_parameters()):
            if name.endswith("scales"):
                value = torch.randint(118, 125, param.shape, dtype = torch.uint8, generator = g)
            else:
                value = torch.randint(0, 256, param.shape, dtype = torch.uint8, generator = g)
            setattr(experts, name, nn.Parameter(value, requires_grad = False))
        experts.finalize()


def test_packed_block_trains_and_matches_its_per_expert_view():
    torch.manual_seed(0)
    mod, _ = _tiny_model("transformers_modules.k3c_dispatch.modeling_tinymoe")
    config = mod.TinyMoeConfig(num_hidden_layers = 1)
    model = mod.TinyMoeForCausalLM(config).to(torch.bfloat16)
    assert swap_in_packed_mxfp4_experts(model, _keys(layers = 1), torch.bfloat16) == ["layers.0.mlp"]
    _materialize_packed(model)
    block = model.layers[0].mlp
    assert is_remote_deepseek_moe(block)
    x = torch.randn(2, 6, H).to(torch.bfloat16)
    block.eval()
    with torch.no_grad():
        port = mod.SparseMoe.moe_infer(block, x.view(-1, H), *block.gate(x))  # loops experts[i]
    assert "SparseMoe" in prepare_remote_moe_for_training(model, verbose = False)
    with torch.no_grad():
        packed_eval = block(x)
    torch.testing.assert_close(packed_eval.view(-1, H).float(), port.float(), atol = 2e-2, rtol = 2e-2)
    block.train()
    xi = x.clone().requires_grad_(True)
    out = block(xi)
    torch.testing.assert_close(out.detach().float(), packed_eval.float(), atol = 0, rtol = 0)
    out.float().square().sum().backward()
    assert xi.grad is not None and xi.grad.abs().sum() > 0
    assert block.gate.weight.grad is not None and block.training and block.gate.training


def _write_checkpoint(root):
    from safetensors.torch import save_file

    d = os.path.join(root, "tiny_mxfp4_moe")
    os.makedirs(d, exist_ok = True)
    with open(os.path.join(d, "modeling_tinymoe.py"), "w") as f:
        f.write(MODELING)
    g = torch.Generator().manual_seed(0)
    tensors = {
        "embed.weight": torch.randn(128, H, generator = g).to(torch.bfloat16),
        "lm_head.weight": (torch.randn(128, H, generator = g) * 0.1).to(torch.bfloat16),
    }
    for key in _keys():
        if key.endswith("proj.weight"):
            tensors[key] = (torch.randn(H, H, generator = g) * 0.1).to(torch.bfloat16)
            continue
        out_f, in_f = (H, I) if ".w2." in key else (I, H)
        if key.endswith("weight_packed"):
            tensors[key] = torch.randint(0, 256, (out_f, in_f // 2), dtype = torch.uint8, generator = g)
        else:
            tensors[key] = torch.randint(
                118, 125, (out_f, in_f // 32), dtype = torch.uint8, generator = g
            )
    for layer in range(2):
        tensors[f"model.layers.{layer}.mlp.gate.weight"] = (
            torch.randn(E, H, generator = g) * 0.1
        ).to(torch.bfloat16)
    # The module tree has no `model.` level: the loader strips the base-model prefix.
    tensors = {k.replace("model.layers.", "layers.", 1): v for k, v in tensors.items()}
    save_file(tensors, os.path.join(d, "model.safetensors"), metadata = {"format": "pt"})
    config = {
        "architectures": ["TinyMoeForCausalLM"],
        "model_type": "tiny_mxfp4_moe",
        "auto_map": {
            "AutoConfig": "modeling_tinymoe.TinyMoeConfig",
            "AutoModelForCausalLM": "modeling_tinymoe.TinyMoeForCausalLM",
        },
        "vocab_size": 128,
        "hidden_size": H,
        "moe_intermediate_size": I,
        "num_experts": E,
        "num_experts_per_tok": K,
        "num_hidden_layers": 2,
        "hidden_act": "situ",
        "torch_dtype": "bfloat16",
        "quantization_config": _mxfp4_plan(),
    }
    json.dump(config, open(os.path.join(d, "config.json"), "w"))
    return d, tensors


def _load(path):
    from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

    config = AutoConfig.from_pretrained(path, trust_remote_code = True)
    assert arm_compressed_tensors_bnb_loading(config, verbose = False) is not None
    return AutoModelForCausalLM.from_pretrained(
        path,
        config = config,
        trust_remote_code = True,
        dtype = torch.bfloat16,
        device_map = {"": 0},
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.bfloat16,
        ),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS),
    reason = "needs compressed-tensors and the transformers 5.8+ loader",
)
def test_remote_code_checkpoint_loads_packed_bytes_verbatim(tmp_path, monkeypatch):
    import bitsandbytes as bnb

    monkeypatch.delenv("UNSLOTH_MXFP4_KEEP_PACKED", raising = False)
    path, tensors = _write_checkpoint(str(tmp_path))
    model = _load(path)
    for layer_index, layer in enumerate(model.layers):
        experts = layer.mlp.experts
        assert isinstance(experts, Mxfp4StackedExperts), type(experts)
        assert isinstance(layer.proj, bnb.nn.Linear4bit)
        prefix = f"layers.{layer_index}.mlp.experts"
        for e in range(E):
            gate_up = experts.gate_up_proj.data[e].reshape(2 * I, H // 2).cpu()
            assert torch.equal(gate_up[:I], tensors[f"{prefix}.{e}.w1.weight_packed"])
            assert torch.equal(gate_up[I:], tensors[f"{prefix}.{e}.w3.weight_packed"])
            down = experts.down_proj.data[e].reshape(H, I // 2).cpu()
            assert torch.equal(down, tensors[f"{prefix}.{e}.w2.weight_packed"])
            scales = experts.gate_up_proj.mxfp4_scales[e].cpu()
            assert torch.equal(scales[:I], tensors[f"{prefix}.{e}.w1.weight_scale"])
            assert torch.equal(
                experts.down_proj.mxfp4_scales[e].cpu(), tensors[f"{prefix}.{e}.w2.weight_scale"]
            )
    assert not any("weight_packed" in n for n, _ in model.named_parameters())
    assert not getattr(model, "_weight_conversions", None) or not any(
        type(op).__name__ == "StackPackedExperts"
        for conv in model._weight_conversions
        for op in getattr(conv, "operations", [])
    )
    ids = torch.randint(0, 128, (2, 8), device = "cuda:0")
    model.eval()
    with torch.no_grad():
        port = model(input_ids = ids).logits  # the port's own loop over experts[i]
    prepare_remote_moe_for_training(model, verbose = False)
    with torch.no_grad():
        grouped = model(input_ids = ids).logits
    torch.testing.assert_close(grouped, port, atol = 5e-2, rtol = 5e-2)
    model.train()
    loss = model(input_ids = ids, labels = ids).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert model.layers[0].mlp.gate.weight.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS),
    reason = "needs compressed-tensors and the transformers 5.8+ loader",
)
def test_keep_packed_off_restores_the_nf4_experts(tmp_path, monkeypatch):
    import bitsandbytes as bnb

    monkeypatch.setenv("UNSLOTH_MXFP4_KEEP_PACKED", "0")
    path, _ = _write_checkpoint(str(tmp_path))
    model = _load(path)
    experts = model.layers[0].mlp.experts
    assert isinstance(experts, nn.ModuleList)
    assert isinstance(experts[0].w1, bnb.nn.Linear4bit)


def test_plan_stacks_the_experts_and_keeps_every_other_packed_linear_all_or_nothing():
    from unsloth.models.compressed_tensors_bnb import plan_mxfp4_keep_packed

    _, model = _tiny_model("transformers_modules.k3s_plan_a.modeling_tinymoe")
    plan = plan_mxfp4_keep_packed(model, _keys())
    assert [name for name, *_ in plan.blocks] == ["layers.0.mlp", "layers.1.mlp"]
    assert plan.linears == []
    # A packed Linear outside the experts stays packed on its own.
    extra = ("model.layers.0.proj.weight_packed", "model.layers.0.proj.weight_scale")
    plan = plan_mxfp4_keep_packed(model, _keys(extra = extra))
    assert len(plan.blocks) == 2 and plan.linears == ["layers.0.proj"]
    # A layer packed only in part, or a packed module with no home: nothing stays packed.
    assert (
        plan_mxfp4_keep_packed(model, _keys(drop = "model.layers.1.mlp.experts.0.w2.weight_packed"))
        is None
    )
    assert plan_mxfp4_keep_packed(model, _keys(extra = ("model.nowhere.weight_packed",))) is None
    assert plan_mxfp4_keep_packed(model, ["model.layers.0.proj.weight"]) is None
    # Experts that are not the plain layout a stack replaces stay packed one Linear each.
    _, model = _tiny_model("transformers_modules.k3s_plan_b.modeling_tinymoe")
    for layer in model.layers:
        for expert in layer.mlp.experts:
            expert.w2 = nn.Linear(I, H, bias = True, device = "meta")
    plan = plan_mxfp4_keep_packed(model, _keys())
    assert plan.blocks == [] and len(plan.linears) == 2 * E * 3


def _packed_expert_linears(model, seed = 0):
    """Every expert Linear as a filled Mxfp4PackedLinear, as the compressed-tensors route adopts them."""
    from unsloth.models.mxfp4_compressed_linear import make_mxfp4_packed_linear

    g = torch.Generator().manual_seed(seed)
    for layer in model.layers:
        for expert in layer.mlp.experts:
            for proj in ("w1", "w2", "w3"):
                old = getattr(expert, proj)
                new = make_mxfp4_packed_linear(
                    old.in_features, old.out_features, dtype = torch.bfloat16
                )
                new.weight_packed.data.copy_(
                    torch.randint(0, 256, new.weight_packed.shape, dtype = torch.uint8, generator = g)
                )
                new.weight_scale.data.copy_(
                    torch.randint(118, 125, new.weight_scale.shape, dtype = torch.uint8, generator = g)
                )
                setattr(expert, proj, new)


def test_compressed_tensors_route_stacks_the_adopted_experts_verbatim():
    from unsloth.models.mxfp4_compressed_linear import stack_packed_expert_linears

    torch.manual_seed(0)
    mod, _ = _tiny_model("transformers_modules.k3s_ct_stack.modeling_tinymoe")
    model = mod.TinyMoeForCausalLM(mod.TinyMoeConfig(num_hidden_layers = 2)).to(torch.bfloat16)
    _packed_expert_linears(model)
    # Snapshots: the per-expert bytes are released as they are stacked.
    before = [
        {
            (e, p): types.SimpleNamespace(
                weight_packed = getattr(x, p).weight_packed.clone(),
                weight_scale = getattr(x, p).weight_scale.clone(),
            )
            for e, x in enumerate(layer.mlp.experts)
            for p in ("w1", "w2", "w3")
        }
        for layer in model.layers
    ]
    x = torch.randn(5, H).to(torch.bfloat16)
    want = [[expert(x) for expert in layer.mlp.experts] for layer in model.layers]
    old_experts = list(model.layers[0].mlp.experts)
    assert stack_packed_expert_linears(model, ["layers.0.mlp", "layers.1.mlp"]) == [
        "layers.0.mlp",
        "layers.1.mlp",
    ]
    assert all("weight_packed" not in x.w1._parameters for x in old_experts)
    for index, layer in enumerate(model.layers):
        experts = layer.mlp.experts
        assert isinstance(experts, Mxfp4StackedExperts)
        for e in range(E):
            gate_up = experts.gate_up_proj.data[e].reshape(2 * I, H // 2)
            assert torch.equal(gate_up[:I], before[index][(e, "w1")].weight_packed)
            assert torch.equal(gate_up[I:], before[index][(e, "w3")].weight_packed)
            assert torch.equal(
                experts.down_proj.data[e].reshape(H, I // 2), before[index][(e, "w2")].weight_packed
            )
            assert torch.equal(
                experts.down_proj.mxfp4_scales[e], before[index][(e, "w2")].weight_scale
            )
            torch.testing.assert_close(experts[e](x), want[index][e], atol = 1e-2, rtol = 1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.skipif(not HAS_CT, reason = "needs compressed-tensors")
def test_remote_code_checkpoint_16bit_load_keeps_packed_stacks(tmp_path, monkeypatch):
    """A 16-bit load (and any load on transformers without the converter hook) goes through
    compressed-tensors' own quantizer: its MXFP4 modules are adopted, then each MoE layer's
    experts are stacked once the weights are in."""
    from transformers import AutoModelForCausalLM
    from unsloth.models.mxfp4_compressed_linear import install_compressed_tensors_keep_packed

    monkeypatch.delenv("UNSLOTH_MXFP4_KEEP_PACKED", raising = False)
    assert install_compressed_tensors_keep_packed()
    path, tensors = _write_checkpoint(str(tmp_path))
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code = True, dtype = torch.bfloat16, device_map = {"": 0}
    )
    for layer_index, layer in enumerate(model.layers):
        experts = layer.mlp.experts
        assert isinstance(experts, Mxfp4StackedExperts), type(experts)
        assert type(layer.proj) is nn.Linear and layer.proj.weight.dtype == torch.bfloat16
        prefix = f"layers.{layer_index}.mlp.experts"
        for e in range(E):
            gate_up = experts.gate_up_proj.data[e].reshape(2 * I, H // 2).cpu()
            assert torch.equal(gate_up[:I], tensors[f"{prefix}.{e}.w1.weight_packed"])
            assert torch.equal(gate_up[I:], tensors[f"{prefix}.{e}.w3.weight_packed"])
            assert torch.equal(
                experts.down_proj.mxfp4_scales[e].cpu(), tensors[f"{prefix}.{e}.w2.weight_scale"]
            )
    assert not any("weight_packed" in n for n, _ in model.named_parameters())
    ids = torch.randint(0, 128, (2, 8), device = "cuda:0")
    model.eval()
    with torch.no_grad():
        port = model(input_ids = ids).logits
    prepare_remote_moe_for_training(model, verbose = False)
    with torch.no_grad():
        grouped = model(input_ids = ids).logits
    torch.testing.assert_close(grouped, port, atol = 5e-2, rtol = 5e-2)


def _saved_state(path):
    from safetensors.torch import load_file

    state = {}
    for name in os.listdir(path):
        if name.endswith(".safetensors"):
            state.update(load_file(os.path.join(path, name)))
    return state


def _packed_tiny_model(name):
    mod, _ = _tiny_model(name)
    config = mod.TinyMoeConfig(num_hidden_layers = 2)
    model = mod.TinyMoeForCausalLM(config).to(torch.bfloat16)
    swap_in_packed_mxfp4_experts(model, _keys(), torch.bfloat16)
    _materialize_packed(model)
    return mod, model


def test_full_save_writes_the_checkpoints_per_expert_keys(tmp_path):
    """A full save of packed (or merged dense) stacks writes `experts.<i>.w1 / w2 / w3.weight`, the
    names the remote code loads, drops the MXFP4 config, and leaves the model as it was."""
    from unsloth_zoo.temporary_patches import mxfp4 as mx

    mx.patch_save_pretrained_mxfp4()
    mod, model = _packed_tiny_model("transformers_modules.k3s_save_a.modeling_tinymoe")
    model.config.quantization_config = _mxfp4_plan()
    stacks = [layer.mlp.experts for layer in model.layers]
    model.save_pretrained(str(tmp_path / "packed"))
    assert [layer.mlp.experts for layer in model.layers] == stacks
    assert model.config.quantization_config == _mxfp4_plan()
    state = _saved_state(str(tmp_path / "packed"))
    assert not any("gate_up" in k or "down_proj" in k or "blocks" in k for k in state)
    assert "quantization_config" not in json.load(open(tmp_path / "packed" / "config.json"))
    for index, layer in enumerate(model.layers):
        gate_up = layer.mlp.experts.gate_up_proj.dequantize(torch.bfloat16).cpu()  # (E, H, 2I)
        down = layer.mlp.experts.down_proj.dequantize(torch.bfloat16).cpu()  # (E, I, H)
        for e in range(E):
            prefix = f"layers.{index}.mlp.experts.{e}"
            assert torch.equal(state[f"{prefix}.w1.weight"], gate_up[e, :, :I].t())
            assert torch.equal(state[f"{prefix}.w3.weight"], gate_up[e, :, I:].t())
            assert torch.equal(state[f"{prefix}.w2.weight"], down[e].t())
    # The remote code's own per-expert loop on the saved weights gives the packed model's output.
    plain = mod.TinyMoeForCausalLM(mod.TinyMoeConfig(num_hidden_layers = 2)).to(torch.bfloat16)
    plain.load_state_dict(state, strict = True)
    ids = torch.randint(0, 128, (2, 8))
    model.eval(), plain.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            plain(input_ids = ids).logits, model(input_ids = ids).logits, atol = 0, rtol = 0
        )

    # A merged adapter leaves a dense stack: written under the same per-expert names.
    experts = model.layers[0].mlp.experts
    dense = experts.gate_up_proj.dequantize(torch.bfloat16) + 0.25
    packed_param = experts.gate_up_proj
    experts.gate_up_proj = nn.Parameter(dense, requires_grad = False)
    model.save_pretrained(str(tmp_path / "merged"))
    state = _saved_state(str(tmp_path / "merged"))
    assert torch.equal(state["layers.0.mlp.experts.1.w3.weight"], dense[1, :, I:].t().cpu())
    experts.gate_up_proj = packed_param

    class Wrapper(nn.Module):
        def __init__(self, base_layer):
            super().__init__()
            self.base_layer = base_layer

    model.layers[1].mlp.experts = Wrapper(model.layers[1].mlp.experts)
    with pytest.raises(RuntimeError, match = "merge_and_unload"):
        model.save_pretrained(str(tmp_path / "wrapped"))


def test_dense_save_config_skips_the_dense_modules_under_bitsandbytes():
    from transformers import BitsAndBytesConfig, PretrainedConfig
    from unsloth_zoo.temporary_patches.mxfp4 import _config_for_dense_save

    config = PretrainedConfig()
    config.quantization_config = BitsAndBytesConfig(
        load_in_4bit = True, llm_int8_skip_modules = ["lm_head"]
    )
    restore = _config_for_dense_save(config, ["layers.0.mlp.experts"])
    assert config.quantization_config.llm_int8_skip_modules == ["lm_head", "layers.0.mlp.experts"]
    restore()
    assert config.quantization_config.llm_int8_skip_modules == ["lm_head"]
    text = PretrainedConfig()
    config = PretrainedConfig(text_config = text)
    config.text_config = text
    config.quantization_config, text.quantization_config = _mxfp4_plan(), _mxfp4_plan()
    restore = _config_for_dense_save(config, [])
    assert not hasattr(config, "quantization_config") and not hasattr(text, "quantization_config")
    restore()
    assert config.quantization_config == text.quantization_config == _mxfp4_plan()


@pytest.mark.skipif(
    not (HAS_CT and HAS_CONVERTERS),
    reason = "needs compressed-tensors and the transformers 5.8+ loader",
)
def test_a_kept_packed_load_registers_no_decompress_op():
    """Every packed module stays packed, so nothing may be routed through the decompressor;
    declined, the decompress catch-all is back."""
    from transformers import BitsAndBytesConfig
    from transformers.quantizers import auto as quantizers_auto
    from unsloth.models.compressed_tensors_bnb import (
        _build_quantization_config,
        install_compressed_tensors_bnb_quantizer,
    )

    assert install_compressed_tensors_bnb_quantizer()
    cls = quantizers_auto.AUTO_QUANTIZER_MAPPING["bitsandbytes_4bit"]
    quantizer = cls(BitsAndBytesConfig(load_in_4bit = True))
    quantizer._unsloth_ct_config = _build_quantization_config(_mxfp4_plan())
    quantizer._unsloth_ct_dtype = torch.bfloat16
    quantizer._unsloth_dtype_plan, quantizer._unsloth_plan_dicts = {}, []
    ops = lambda convs: [type(op).__name__ for c in convs for op in getattr(c, "operations", [])]  # noqa: E731
    quantizer._unsloth_keep_packed, quantizer._unsloth_packed_experts = True, ["layers.0.mlp"]
    kept = ops(quantizer.update_weight_conversions([]))
    assert kept.count("StackPackedExperts") == 4 and "DecompressPackedWeights" not in kept
    quantizer._unsloth_keep_packed, quantizer._unsloth_packed_experts = False, []
    assert "DecompressPackedWeights" in ops(quantizer.update_weight_conversions([]))


def _peft_target_parameters(model, monkeypatch, **kwargs):
    """The `target_parameters` FastBaseModel.get_peft_model hands PEFT, captured at LoraConfig."""
    import functools
    import unsloth.models.vision as vision

    class _Captured(Exception):
        pass

    real = vision.LoraConfig

    @functools.wraps(real)
    def capture(**config):
        raise _Captured(config.get("target_parameters"))

    monkeypatch.setattr(vision, "LoraConfig", capture)
    with pytest.raises(_Captured) as captured:
        vision.FastBaseModel.get_peft_model(model, r = 4, **kwargs)
    return captured.value.args[0]


@pytest.mark.parametrize(
    "flags, experts",
    [
        ({}, True),
        ({"finetune_mlp_modules": False}, False),
        ({"finetune_language_layers": False}, False),
    ],
)
def test_packed_expert_targets_follow_the_finetune_family_flags(flags, experts, monkeypatch):
    mod, _ = _tiny_model("transformers_modules.k3s_scope.modeling_tinymoe")
    model = mod.TinyMoeForCausalLM(mod.TinyMoeConfig(num_hidden_layers = 1)).to(torch.bfloat16)
    swap_in_packed_mxfp4_experts(model, _keys(layers = 1), torch.bfloat16)
    _materialize_packed(model)
    model.max_seq_length = 64
    # Something left to train when a family is scoped out.
    model.vision_tower = nn.Module()
    model.vision_tower.attn = nn.Module()
    model.vision_tower.attn.q_proj = nn.Linear(H, H, bias = False)
    got = _peft_target_parameters(model, monkeypatch, target_modules = ["q_proj", "w1", "w2"], **flags)
    want = ["experts.gate_up_proj", "experts.down_proj"] if experts else []
    assert sorted(got or []) == sorted(want)


def test_nothing_stays_packed_without_the_zoo_full_save_support(monkeypatch):
    """A kept-packed model saves loadable checkpoints only through unsloth_zoo's save patch; an
    unsloth_zoo without it keeps the previous route instead."""
    from unsloth_zoo.temporary_patches import mxfp4 as zoo_mxfp4

    monkeypatch.delenv("UNSLOTH_MXFP4_KEEP_PACKED", raising = False)
    assert keep_mxfp4_experts_packed(_mxfp4_plan())
    monkeypatch.delattr(zoo_mxfp4, "_densified_module_names")
    assert not keep_mxfp4_experts_packed(_mxfp4_plan())


# A training-capable remote MoE (DeepSeek-V2/V3 style): its own `if self.training:` branch loops
# over the experts, so the remote MoE shim leaves it alone.
TRAINING_MODELING = MODELING.replace(
    """        if not self.training:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight)
        else:
            raise NotImplementedError("inference only")""",
    """        if self.training:
            flat = topk_idx.view(-1)
            x = hidden_states.repeat_interleave(topk_idx.shape[1], dim=0)
            y = torch.empty_like(x)
            for i, expert in enumerate(self.experts):
                y[flat == i] = expert(x[flat == i]).to(y.dtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1).to(y.dtype)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight)""",
).replace("        assert not self.training\n", "")


def test_only_blocks_the_remote_moe_shim_dispatches_are_stacked():
    """A stack runs only through the shim's dispatch: a block the shim leaves alone (its own
    training branch, or expert parallel) would index or iterate the stack itself, which breaks
    under an expert LoRA wrapper. Such experts stay packed one Linear each instead."""
    from unsloth.models.compressed_tensors_bnb import plan_mxfp4_keep_packed

    assert TRAINING_MODELING != MODELING
    _, model = _tiny_model("transformers_modules.k3s_train_branch.modeling_tinymoe", source = TRAINING_MODELING)
    assert not is_remote_deepseek_moe(model.layers[0].mlp)
    plan = plan_mxfp4_keep_packed(model, _keys())
    assert plan.blocks == [] and len(plan.linears) == 2 * E * 3
    _, model = _tiny_model("transformers_modules.k3s_ep.modeling_tinymoe")
    model.layers[1].mlp.ep_size = 2
    plan = plan_mxfp4_keep_packed(model, _keys())
    assert plan.blocks == [] and len(plan.linears) == 2 * E * 3
