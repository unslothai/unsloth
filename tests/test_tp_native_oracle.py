# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor parallel plan derivation, refusals, and a simulated-TP oracle (summed shard copies), fp32 CPU."""

import copy
import importlib.util
import os
import sys
import types

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_tp():
    """tp_native.py without importing the unsloth package (which needs a GPU)."""
    name = "unsloth.models.tp_native"
    if name in sys.modules:
        return sys.modules[name]
    import importlib.machinery

    pkg = types.ModuleType("unsloth")
    pkg.__path__ = [os.path.join(REPO, "unsloth")]
    models = types.ModuleType("unsloth.models")
    models.__path__ = [os.path.join(REPO, "unsloth", "models")]
    for package in (pkg, models):
        # A spec, so a later importlib.util.find_spec("unsloth") in the same session does not raise.
        package.__spec__ = importlib.machinery.ModuleSpec(package.__name__, None, is_package = True)
        package.__spec__.submodule_search_locations = package.__path__
    sys.modules.setdefault("unsloth", pkg)
    sys.modules.setdefault("unsloth.models", models)
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO, "unsloth", "models", "tp_native.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tpn = _load_tp()


def skip_unsupported(arch):
    """transformers 4.x StableLmAttention reshapes with hidden_size (the global width): apply refuses it."""
    import transformers
    if arch == "stablelm" and int(transformers.__version__.split(".")[0]) < 5:
        pytest.skip("transformers 4.x StableLM attention is refused (test_stablelm_4x_is_refused)")


LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
QWEN3 = "trl-internal-testing/tiny-Qwen3ForCausalLM"
GEMMA2 = "trl-internal-testing/tiny-Gemma2ForCausalLM"
GEMMA3 = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
MISTRAL = "hf-internal-testing/tiny-random-MistralForCausalLM"
PHI3 = "trl-internal-testing/tiny-Phi3ForCausalLM-3"
QWEN2 = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5-Coder"
STABLELM = "hf-internal-testing/tiny-random-StableLmForCausalLM"  # no transformers TP plan
OLMO2 = "hf-internal-testing/tiny-random-Olmo2ForCausalLM"
QWEN3_MOE = "trl-internal-testing/tiny-Qwen3MoeForCausalLM"
DEEPSEEK_V3 = "trl-internal-testing/tiny-DeepseekV3ForCausalLM"

# Archs that shard at tp = 2 (config overrides make every count divisible where the tiny repo is not).
ARCHS = {
    "llama": (LLAMA, {}),
    "llama_bias": (LLAMA, {"attention_bias": True, "mlp_bias": True}),  # row biases: added once
    "qwen3": (QWEN3, {}),
    "gemma2": (GEMMA2, {}),
    "gemma3": (GEMMA3, {"num_attention_heads": 4, "num_key_value_heads": 2}),
    "mistral": (MISTRAL, {"intermediate_size": 64}),
    "phi3": (PHI3, {}),
    "qwen2": (QWEN2, {}),
    "stablelm": (STABLELM, {"use_qkv_bias": True, "intermediate_size": 64}),
}


def build(
    repo,
    overrides = None,
    lora = True,
    seed = 0,
    train_norms = False,
    **lora_kwargs,
):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(repo)
    text = config.get_text_config()
    for key, value in (overrides or {}).items():
        setattr(text, key, value)
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("bias"):
                p.normal_(0, 0.1)  # random biases: their placement matters
            elif "norm" in name and p.ndim == 1:
                p.add_(torch.randn_like(p) * 0.1)
    if not lora:
        return model.eval()
    from peft import LoraConfig, get_peft_model

    kwargs = dict(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)
    kwargs.update(lora_kwargs)
    torch.manual_seed(seed + 1)
    model = get_peft_model(model, LoraConfig(**kwargs))
    if train_norms:
        for name, p in model.named_parameters():
            if name.endswith(("q_norm.weight", "k_norm.weight")):
                p.requires_grad_(True)
    return model.eval()


def _batch(
    model,
    seed = 1,
    bsz = 2,
    seq = 7,
):
    vocab = tpn._text_config(model).vocab_size
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (bsz, seq), generator = g)


class _SimRegion(torch.nn.Module):
    """tp shard copies of one region, outputs summed (reduce_from_tp) plus the row bias added once."""

    def __init__(self, copies, row_bias):
        super().__init__()
        self.copies = torch.nn.ModuleList(copies)
        self.row_bias = row_bias

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["copies"][0], name)

    def forward(self, *args, **kwargs):
        outs = [c(*args, **kwargs) for c in self.copies]
        tuple_out = isinstance(outs[0], tuple)
        y = sum(o[0] if tuple_out else o for o in outs)
        if self.row_bias is not None:
            y = y + self.row_bias
        return (y,) + tuple(outs[0][1:]) if tuple_out else y


def _shard_copy(region, rank, tp):
    module = copy.deepcopy(region.module)
    for proj in region.projs:
        child = getattr(module, proj.name)
        weight, base = tpn._weight(child)
        col = proj.style == tpn.COLWISE
        in_parts = (weight.shape[1],)
        with torch.no_grad():
            if col:
                base.weight = torch.nn.Parameter(
                    tpn.shard_slice(weight, 0, proj.parts, rank, tp), False
                )
                if base.bias is not None:
                    base.bias = torch.nn.Parameter(
                        tpn.shard_slice(base.bias, 0, proj.parts, rank, tp), False
                    )
            else:
                base.weight = torch.nn.Parameter(
                    tpn.shard_slice(weight, 1, in_parts, rank, tp), False
                )
                base.bias = None
            for adapter in getattr(child, "lora_A", {}):
                A, B = child.lora_A[adapter], child.lora_B[adapter]
                if col:
                    B.weight = torch.nn.Parameter(
                        tpn.shard_slice(B.weight, 0, proj.parts, rank, tp)
                    )
                else:
                    A.weight = torch.nn.Parameter(tpn.shard_slice(A.weight, 1, in_parts, rank, tp))
    if region.kind == "attention":
        tpn._set_local_heads(module, region, tp)
    return module


def simulate(
    model,
    tp,
    user_plan = None,
):
    """A copy of `model` whose regions run as tp summed shard copies, and the plan used."""
    sim = copy.deepcopy(model)
    plan = tpn.derive_tp_plan(sim, tp, user_plan)
    for region in plan.regions:
        parent_path, name = region.path.rsplit(".", 1)
        parent = tpn._unwrap(sim).get_submodule(parent_path)
        row_bias = None
        for proj in region.projs:
            base = tpn._weight(proj.module)[1]
            if proj.style == tpn.ROWWISE and base.bias is not None:
                row_bias = base.bias.detach()
        copies = [_shard_copy(region, r, tp) for r in range(tp)]
        setattr(parent, name, _SimRegion(copies, row_bias))
    return sim, plan


def _sim_grads(sim, plan, tp):
    """{param path relative to the unwrapped model: full gradient} reassembled from the shard copies."""
    grads = {}
    inner = tpn._unwrap(sim)
    for region in plan.regions:
        sim_region = inner.get_submodule(region.path)
        for proj in region.projs:
            col = proj.style == tpn.COLWISE
            children = [getattr(c, proj.name) for c in sim_region.copies]
            for adapter in getattr(children[0], "lora_A", {}):
                As = [c.lora_A[adapter].weight.grad for c in children]
                Bs = [c.lora_B[adapter].weight.grad for c in children]
                prefix = f"{region.path}.{proj.name}"
                if col:
                    grads[f"{prefix}.lora_A.{adapter}.weight"] = sum(As)
                    grads[f"{prefix}.lora_B.{adapter}.weight"] = tpn._unpermute(
                        torch.cat(Bs, 0), 0, proj.parts, tp
                    )
                else:
                    grads[f"{prefix}.lora_A.{adapter}.weight"] = torch.cat(As, 1)
                    grads[f"{prefix}.lora_B.{adapter}.weight"] = sum(Bs)
        for norm_name, _ in region.norms:
            for pname, _ in getattr(sim_region.copies[0], norm_name).named_parameters():
                gs = [getattr(c, norm_name).get_parameter(pname).grad for c in sim_region.copies]
                if gs[0] is not None:
                    grads[f"{region.path}.{norm_name}.{pname}"] = sum(gs)
    return grads


def _ref_grads(model):
    inner = tpn._unwrap(model)
    return {n: p.grad for n, p in inner.named_parameters() if p.grad is not None}


def _loss(model, ids):
    out = model(input_ids = ids, labels = ids, use_cache = False)
    return out.loss, out.logits


@pytest.mark.parametrize("arch", sorted(ARCHS))
def test_simulated_tp_matches_unsharded(arch):
    skip_unsupported(arch)
    repo, overrides = ARCHS[arch]
    model = build(repo, overrides, train_norms = arch == "qwen3")
    ids = _batch(model)
    ref_loss, ref_logits = _loss(model, ids)
    ref_loss.backward()
    ref = _ref_grads(model)
    tp = 2
    sim, plan = simulate(model, tp)
    for p in sim.parameters():
        p.grad = None
    loss, logits = _loss(sim, ids)
    loss.backward()
    got = _sim_grads(sim, plan, tp)
    torch.testing.assert_close(logits, ref_logits, rtol = 1e-5, atol = 1e-5)
    torch.testing.assert_close(loss, ref_loss, rtol = 1e-6, atol = 1e-6)
    lora = [k for k in ref if "lora_" in k]
    assert lora and set(lora) <= set(got), sorted(set(lora) - set(got))[:4]
    for key in ref:
        if key in got:
            torch.testing.assert_close(got[key], ref[key], rtol = 1e-4, atol = 1e-6, msg = key)
    if arch == "qwen3":
        assert any(
            k.endswith("q_norm.weight") for k in got
        )  # the per-head norm grad sum was compared


@pytest.mark.parametrize("tp", [2, 4])
def test_simulated_tp_gqa_widths(tp):
    model = build(
        QWEN3, {"num_attention_heads": 8, "num_key_value_heads": 4, "intermediate_size": 32}
    )
    ids = _batch(model)
    ref_loss, ref_logits = _loss(model, ids)
    sim, _ = simulate(model, tp)
    _, logits = _loss(sim, ids)
    torch.testing.assert_close(logits, ref_logits, rtol = 1e-5, atol = 1e-5)


def test_mutation_unsummed_replicated_factor_fails():
    """Dropping the tp sum of a replicated factor's gradient (taking one copy's partial) is caught."""
    model = build(LLAMA)
    ids = _batch(model)
    _loss(model, ids)[0].backward()
    ref = _ref_grads(model)
    sim, plan = simulate(model, 2)
    _loss(sim, ids)[0].backward()
    region = plan.regions[0]
    proj = region.projs[0]
    child = getattr(tpn._unwrap(sim).get_submodule(region.path).copies[0], proj.name)
    key = f"{region.path}.{proj.name}.lora_A.default.weight"
    with pytest.raises(AssertionError):
        torch.testing.assert_close(
            child.lora_A["default"].weight.grad, ref[key], rtol = 1e-4, atol = 1e-6
        )


def _styles(plan):
    return {k.split(".", 3)[-1] if k.count(".") >= 3 else k: v for k, v in plan.styles().items()}


@pytest.mark.parametrize("arch", sorted(ARCHS))
def test_plan_covers_every_layer(arch):
    skip_unsupported(arch)
    repo, overrides = ARCHS[arch]
    model = build(repo, overrides)
    plan = tpn.derive_tp_plan(model, 2)
    layers = tpn._text_config(model).num_hidden_layers
    assert len(plan.regions) == 2 * layers
    assert plan.source == ("structural" if arch == "stablelm" else "transformers")
    styles = {(p.name, p.style) for r in plan.regions for p in r.projs}
    if arch == "phi3":
        assert styles == {
            ("qkv_proj", "colwise"),
            ("o_proj", "rowwise"),
            ("gate_up_proj", "colwise"),
            ("down_proj", "rowwise"),
        }
        parts = {p.name: p.parts for r in plan.regions for p in r.projs}
        text = tpn._text_config(model)
        hd = text.hidden_size // text.num_attention_heads
        assert parts["qkv_proj"] == (
            text.num_attention_heads * hd,
            text.num_key_value_heads * hd,
            text.num_key_value_heads * hd,
        )
        assert parts["gate_up_proj"] == (text.intermediate_size, text.intermediate_size)
    else:
        assert styles == {
            ("q_proj", "colwise"),
            ("k_proj", "colwise"),
            ("v_proj", "colwise"),
            ("o_proj", "rowwise"),
            ("gate_proj", "colwise"),
            ("up_proj", "colwise"),
            ("down_proj", "rowwise"),
        }
    norms = {n for r in plan.regions for n, _ in r.norms}
    assert norms == ({"q_norm", "k_norm"} if arch in ("qwen3", "gemma3") else set())


def test_structural_plan_equals_transformers_plan():
    model = build(QWEN3)
    hf = tpn.derive_tp_plan(model, 2)
    text = tpn._text_config(model)
    text.base_model_tp_plan = None
    for module in model.modules():
        if "_tp_plan" in module.__dict__:
            module._tp_plan = None
    structural = tpn.derive_tp_plan(model, 2)
    assert structural.source == "structural"
    assert structural.styles() == hf.styles()
    assert [len(r.norms) for r in structural.regions] == [len(r.norms) for r in hf.regions]


def test_user_plan_overrides():
    model = build(LLAMA, lora = False)
    plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    assert tpn.derive_tp_plan(model, 2, plan).source == "user"
    with pytest.raises(ValueError, match = "unknown tensor parallel style"):
        tpn.derive_tp_plan(model, 2, {**plan, "layers.*.mlp.down_proj": "diagonal"})
    # A plan entry wins over the name rule; names fill what the plan leaves out.
    swapped = tpn.derive_tp_plan(model, 2, {**plan, "layers.*.mlp.up_proj": "rowwise"}).styles()
    assert swapped["model.layers.0.mlp.up_proj"] == "rowwise"
    missing = {k: v for k, v in plan.items() if "down_proj" not in k}
    assert (
        tpn.derive_tp_plan(model, 2, missing).styles()
        == tpn.derive_tp_plan(model, 2, plan).styles()
    )


def test_unknown_module_inside_region_is_refused():
    model = build(LLAMA, lora = False)
    model.model.layers[1].mlp.gate_scale = torch.nn.Linear(4, 4)
    with pytest.raises(ValueError, match = "no rule for 'model.layers.1.mlp.gate_scale'"):
        tpn.derive_tp_plan(model, 2)
    model = build(LLAMA, lora = False)
    model.model.layers[0].self_attn.sinks = torch.nn.Parameter(torch.zeros(4))
    with pytest.raises(ValueError, match = "owns the parameter 'sinks'"):
        tpn.derive_tp_plan(model, 2)


def test_mla_config_refused():
    model = build(LLAMA, lora = False)
    tpn._text_config(model).kv_lora_rank = 16
    with pytest.raises(ValueError, match = "latent attention"):
        tpn.derive_tp_plan(model, 2)


def test_local_heads_set_per_module_not_config():
    model = build(PHI3)
    text = tpn._text_config(model)
    heads, kv = text.num_attention_heads, text.num_key_value_heads
    sim, plan = simulate(model, 2)
    attn = tpn._unwrap(sim).get_submodule(plan.regions[0].path).copies[0]
    assert attn._unsloth_tp_heads == heads // 2 and attn._unsloth_tp_kv_heads == kv // 2
    assert attn.config.num_attention_heads == heads // 2  # Phi3 slices qkv with config heads
    assert tpn._text_config(sim).num_attention_heads == heads  # shared config untouched
    assert tpn._text_config(model).num_attention_heads == heads


def test_shard_slice_unpermute_round_trip():
    t = torch.arange(10 * 3).view(10, 3)
    parts = (6, 2, 2)
    shards = [tpn.shard_slice(t, 0, parts, r, 2) for r in range(2)]
    assert shards[0][:3].tolist() == t[:3].tolist() and shards[0][3].tolist() == t[6].tolist()
    assert torch.equal(tpn._unpermute(torch.cat(shards), 0, parts, 2), t)


@pytest.mark.parametrize(
    "repo,overrides,tp,match",
    [
        (LLAMA, {"num_attention_heads": 4, "num_key_value_heads": 4}, 3, "num_attention_heads = 4"),
        (QWEN3, {"num_attention_heads": 6, "num_key_value_heads": 3}, 2, "num_key_value_heads = 3"),
        (GEMMA3, {}, 2, "larger than num_key_value_heads = 1"),
        (MISTRAL, {}, 2, "intermediate_size = 37"),
        (OLMO2, {}, 2, "normalizes .* features, not one head"),
        (QWEN3_MOE, {}, 2, "mixture-of-experts"),
        (DEEPSEEK_V3, {}, 2, "mixture-of-experts|latent attention"),
    ],
)
def test_refusals_plan(repo, overrides, tp, match):
    model = build(repo, overrides, lora = False)
    with pytest.raises(ValueError, match = match):
        tpn.derive_tp_plan(model, tp)


def test_refuses_attention_and_layer_dropout():
    with pytest.raises(ValueError, match = "attention dropout"):
        tpn.derive_tp_plan(build(LLAMA, {"attention_dropout": 0.1}), 2)
    model = build(LLAMA)
    model.get_base_model().model.layers[0].mlp.add_module("drop", torch.nn.Dropout(0.1))
    with pytest.raises(ValueError, match = "dropout inside decoder layers"):
        tpn.derive_tp_plan(model, 2)


def test_refuses_dora_and_lora_bias():
    for kwargs, match in (
        ({"use_dora": True}, "DoRA"),
        ({"lora_bias": True}, "lora_bias"),
        ({"lora_dropout": 0.1}, "lora_dropout"),
    ):
        model = build(LLAMA, **kwargs)
        region = tpn.derive_tp_plan(model, 2).regions[0]
        with pytest.raises(ValueError, match = match):
            tpn._refuse_adapter(region.path, region.projs[0].module)


def test_refuses_quantized_and_trainable_bases():
    model = build(LLAMA)
    region = tpn.derive_tp_plan(model, 2).regions[0]
    proj = region.projs[0].module
    weight, base = tpn._weight(proj)
    with pytest.raises(ValueError, match = "fp8"):
        tpn._refuse_base("p", proj, weight.to(torch.float8_e4m3fn), base)
    int8 = type("Int8Params", (torch.nn.Parameter,), {})
    with pytest.raises(ValueError, match = "8-bit"):
        tpn._refuse_base("p", proj, int8(weight.detach()), base)
    gptq = type("QuantLinear", (torch.nn.Module,), {})()
    with pytest.raises(ValueError, match = "GPTQ"):
        tpn._refuse_base("p", proj, weight, gptq)
    with pytest.raises(ValueError, match = "full finetuning"):
        tpn._refuse_base("p", proj, torch.nn.Parameter(weight.detach().clone()), base)


def test_refuses_global_width_attention():
    model = build(LLAMA, lora = False)
    attn = model.model.layers[0].self_attn

    def forward(self, hidden_states, **kwargs):
        return hidden_states.reshape(1, 1, self.config.hidden_size), None

    attn.forward = types.MethodType(forward, attn)
    with pytest.raises(ValueError, match = "hidden_size"):
        tpn.derive_tp_plan(model, 2)


def test_stablelm_4x_is_refused():
    import transformers

    if int(transformers.__version__.split(".")[0]) >= 5:
        pytest.skip("transformers 5 StableLM attention follows the local heads")
    repo, overrides = ARCHS["stablelm"]
    with pytest.raises(ValueError, match = "reshapes with hidden_size"):
        tpn.derive_tp_plan(build(repo, overrides), 2)


# Phi3's fused qkv under a plan that calls it packed: the layout is still [q, kv, kv] (GQA: unequal parts).
PHI3_PACKED_PLAN = {
    "layers.*.self_attn.qkv_proj": "packed_colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_up_proj": "packed_colwise",
    "layers.*.mlp.down_proj": "rowwise",
}


def test_fused_qkv_under_a_packed_plan_keeps_the_gqa_layout():
    model = build(PHI3)
    text = tpn._text_config(model)
    assert text.num_key_value_heads < text.num_attention_heads  # the tiny Phi3 is GQA (4 / 2)
    hd = text.hidden_size // text.num_attention_heads
    q, kv = text.num_attention_heads * hd, text.num_key_value_heads * hd
    ids = _batch(model)
    _, ref_logits = _loss(model, ids)
    sim, plan = simulate(model, 2, PHI3_PACKED_PLAN)
    parts = {p.name: p.parts for p in plan.regions[0].projs}
    assert parts["qkv_proj"] == (q, kv, kv), parts
    _, logits = _loss(sim, ids)
    torch.testing.assert_close(logits, ref_logits, rtol = 1e-5, atol = 1e-5)
    with pytest.raises(ValueError, match = "has no known layout"):
        tpn.derive_tp_plan(
            build(LLAMA, lora = False), 2, {"layers.*.self_attn.q_proj": "packed_colwise"}
        )


class _FakeLora(torch.nn.Module):
    def __init__(self, active):
        super().__init__()
        self.base_layer = torch.nn.Linear(4, 6, bias = False)
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 2, bias = False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 6, bias = False)})
        self.scaling = {"default": 2.0}
        self.disable_adapters = False
        self.merged = False
        if active is not None:
            self.active_adapters = active


def test_active_adapter_matches_get_lora_parameters():
    singular = _FakeLora(None)
    singular.active_adapter = "default"
    assert tpn._active_adapter(singular) == ["default"]
    assert tpn._active_adapter(_FakeLora(None)) == ["default"]
    assert tpn._active_adapter(_FakeLora("default")) == ["default"]
    assert tpn._active_adapter(_FakeLora([])) == []
    for active in (None, "default", ["default"]):
        W, _, A, B, s = tpn._tp_lora_parameters(_FakeLora(active))
        assert A is not None and B is not None and s == 2.0
    for active in ([], ["other"]):
        assert tpn._tp_lora_parameters(_FakeLora(active))[2:] == (None, None, None)
    with pytest.raises(ValueError, match = "single active adapter"):
        tpn._tp_lora_parameters(_FakeLora(["default", "other"]))


def test_refuses_forward_qat_and_activation_fake_quantizers():
    model = build(LLAMA)
    region = tpn.derive_tp_plan(model, 2).regions[0]
    proj = region.projs[0].module
    weight, base = tpn._weight(proj)
    # torchao's NVFP4 / MX QAT linears fake-quantize inside forward() and carry no weight_fake_quantizer.
    forward_qat = type("NVFP4FakeQuantizedLinear", (torch.nn.Linear,), {})(4, 4).requires_grad_(
        False
    )
    with pytest.raises(ValueError, match = "QAT fake quantization"):
        tpn._refuse_base(
            "p", torch.nn.ModuleDict({"base_layer": forward_qat}), forward_qat.weight, forward_qat
        )
    try:
        from torchao.prototype import qat
        real_classes = [
            (qat.MXFakeQuantizedLinear, qat.MXFakeQuantizeConfig),
            (qat.NVFP4FakeQuantizedLinear, qat.NVFP4FakeQuantizeConfig),
        ]
    except Exception:
        real_classes = []
    for cls, cfg in real_classes:  # the real torchao classes (weights frozen, as under LoRA QAT)
        real = cls(32, 32, activation_config = cfg(), weight_config = cfg()).requires_grad_(False)
        assert getattr(real, "weight_fake_quantizer", None) is None
        with pytest.raises(ValueError, match = "QAT fake quantization"):
            tpn._refuse_base("p", torch.nn.ModuleDict({"base_layer": real}), real.weight, real)
    proj.lora_A["default"].activation_fake_quantizer = torch.nn.Identity()
    with pytest.raises(ValueError, match = "QAT fake quantization"):
        tpn._refuse_base("p", proj, weight, base)
    del proj.lora_A["default"].activation_fake_quantizer
    tpn._refuse_base("p", proj, weight, base)  # plain LoRA passes


def test_refuses_modules_to_save_inside_a_region():
    model = build(
        LLAMA,
        target_modules = ["k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save = ["q_proj"],
    )
    region = tpn.derive_tp_plan(model, 2).regions[0]
    q = next(p for p in region.projs if p.name == "q_proj")
    with pytest.raises(ValueError, match = "modules_to_save"):
        tpn._refuse_adapter(f"{region.path}.q_proj", q.module)


def test_region_hook_takes_the_forward_input_by_name():
    model = build(LLAMA, lora = False)
    mlp = model.model.layers[0].mlp
    mlp._unsloth_tp_group = (
        None  # copy_to_tp's forward is the identity; the group is read in backward
    )
    mlp._unsloth_tp_input = "x"  # what apply_tensor_parallel records for LlamaMLP.forward(self, x)
    h = torch.randn(1, 2, 16)
    args, kwargs = tpn._region_pre_hook(mlp, (), {"x": h})  # mlp(x = h)
    assert args == () and torch.equal(kwargs["x"], h) and kwargs["x"] is not h
    attn = model.model.layers[0].self_attn
    assert tpn._input_name(mlp) == "x" and tpn._input_name(attn) == "hidden_states"
    args, kwargs = tpn._region_pre_hook(mlp, (h,), {})
    assert torch.equal(args[0], h)
    with pytest.raises(RuntimeError, match = "got no tensor input"):
        tpn._region_pre_hook(mlp, (), {"y": h})


def test_clip_takes_a_single_parameter():
    a = torch.nn.Parameter(torch.randn(5, 3))
    b = torch.nn.Parameter(a.detach().clone())
    a.grad = torch.randn(5, 3) * 4
    b.grad = a.grad.clone()
    ref = torch.nn.utils.clip_grad_norm_(b, 1.0)
    got = tpn.tp_clip_grad_norm_(a, 1.0)
    torch.testing.assert_close(got, ref)
    torch.testing.assert_close(a.grad, b.grad)


def _load_flex():
    name = "_tp_flex_attention"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(REPO, "unsloth", "kernels", "flex_attention.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


def test_gemma2_softcapping_helpers_use_local_heads():
    """Gemma2 softcapping helpers honour _unsloth_tp_heads over the config's global counts."""
    flex = _load_flex()
    import types as _types

    def attn(
        heads,
        kv,
        tp_heads = None,
        tp_kv = None,
    ):
        config = _types.SimpleNamespace(
            num_attention_heads = heads,
            num_key_value_heads = kv,
            query_pre_attn_scalar = 16,
            attn_logit_softcapping = 50.0,
        )
        m = torch.nn.Module()
        m.config, m.head_dim, m.num_key_value_groups = config, 8, heads // kv
        if tp_heads is not None:
            m._unsloth_tp_heads, m._unsloth_tp_kv_heads = tp_heads, tp_kv
        return m

    bsz, q_len = 2, 5
    g = torch.Generator().manual_seed(0)
    Q = torch.randn(bsz, 2, q_len, 8, generator = g)
    K = torch.randn(bsz, 1, q_len, 8, generator = g)
    V = torch.randn(bsz, 1, q_len, 8, generator = g)
    mask = torch.triu(torch.full((q_len, q_len), float("-inf")), 1)
    local, tp_rank = attn(2, 1), attn(4, 2, 2, 1)
    eager = flex._compiled_slow_attention_softcapping._torchdynamo_orig_callable
    for fn in (eager, flex.slow_inference_attention_softcapping):
        want = fn(Q, K, V, mask, local, bsz, q_len)
        got = fn(Q, K, V, mask, tp_rank, bsz, q_len)
        assert got.shape == (bsz, q_len, 2 * 8)
        torch.testing.assert_close(got, want)


def test_stub_packages_keep_valid_specs():
    # Later tests in the same session (e.g. the fused-loss harness) call find_spec("unsloth").
    for name in ("unsloth", "unsloth.models"):
        assert sys.modules[name].__spec__ is not None, name
    importlib.util.find_spec("unsloth.models")


def test_generated_trl_trainer_init_skips_for_inference_on_a_tp_model():
    """The generated TRL trainers end __init__ with model.for_inference(); a tensor parallel trainer
    (tp_native.tp_trainer_setup sets _unsloth_tensor_parallel) must skip it, every other trainer keeps it."""
    import re
    import textwrap

    source = open(os.path.join(REPO, "unsloth", "models", "rl.py"), encoding = "utf-8").read()
    match = re.search(
        r"super\(\)\.__init__\(\{RLTrainer_call_args\}\{RLTrainer_kwargs\}\)\n(.*?)\n\{RLTrainer_post\}",
        source,
        re.S,
    )
    assert match, "generated trainer __init__ tail not found"
    tail = textwrap.dedent(match.group(1))
    calls = []

    class Model:
        def for_inference(self):
            calls.append(type(self).__name__)

    for tensor_parallel in (False, True):
        trainer = types.SimpleNamespace(
            **({"_unsloth_tensor_parallel": True} if tensor_parallel else {})
        )
        exec(tail, {}, {"self": trainer, "model": Model()})
    assert calls == ["Model"], calls


def test_trainer_dataloader_workers_are_seeded_by_dp_rank():
    import functools

    pytest.importorskip("accelerate")
    from accelerate import Accelerator
    from transformers.trainer_utils import seed_worker

    tp = _load_tp()
    tp._install_tp_accelerate_hooks()
    accelerator = Accelerator(cpu = True)
    accelerator.__dict__[tp._TP_DATA] = (1, 0)  # dp 1: every tp rank is dp rank 0
    try:
        init_fn = functools.partial(seed_worker, num_workers = 2, rank = 3)
        loader = torch.utils.data.DataLoader(
            list(range(8)), batch_size = 2, num_workers = 2, worker_init_fn = init_fn
        )
        prepared = accelerator.prepare_data_loader(loader)
    finally:
        accelerator.__dict__.pop(tp._TP_DATA, None)
    assert prepared.worker_init_fn.func is seed_worker
    assert prepared.worker_init_fn.keywords == {"num_workers": 2, "rank": 0}
