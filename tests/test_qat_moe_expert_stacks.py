# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""qat_scheme on packed 3-D MoE expert stacks (gpt-oss; Qwen3-MoE on transformers 5.x): the expert matmuls see
fake-quantized weights (trainable: every forward, straight-through gradient; frozen: rounded once), CPU."""

import importlib.util

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None, reason = "require torchao"
)


def _tiny_moe(arch, intermediate = 128):
    import transformers

    torch.manual_seed(0)
    if arch == "gpt_oss":
        cfg = transformers.GptOssConfig(
            vocab_size = 256, hidden_size = 128, intermediate_size = intermediate, num_hidden_layers = 2,
            num_attention_heads = 4, num_key_value_heads = 2, head_dim = 32, num_local_experts = 4,
            num_experts_per_tok = 2, layer_types = ["full_attention"] * 2,
        )  # fmt: skip
    else:
        mod = pytest.importorskip("transformers.models.qwen3_moe.modeling_qwen3_moe")
        if not hasattr(mod, "Qwen3MoeExperts"):
            pytest.skip("transformers 4.x keeps Qwen3-MoE experts as per-expert Linears")
        cfg = transformers.Qwen3MoeConfig(
            vocab_size = 256, hidden_size = 128, intermediate_size = 256, moe_intermediate_size = intermediate,
            num_hidden_layers = 2, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 32,
            num_experts = 4, num_experts_per_tok = 2, decoder_sparse_step = 1, mlp_only_layers = [],
        )  # fmt: skip
    return transformers.AutoModelForCausalLM.from_config(cfg, torch_dtype = torch.float32)


def _stacks(model):
    return {
        f"{n}.{a}": p
        for n, m in model.named_modules()
        for a in ("gate_up_proj", "down_proj")
        if (p := m._parameters.get(a)) is not None and p.dim() == 3
    }


def _reference_int4(w, transposed):
    """Independent reference: torchao's int4 weight fake quantizer applied one expert at a time."""
    from torchao.quantization import Int4WeightOnlyConfig
    from torchao.quantization.qat.fake_quantize_config import _infer_fake_quantize_configs
    from torchao.quantization.qat.fake_quantizer import FakeQuantizerBase

    q = FakeQuantizerBase.from_config(
        _infer_fake_quantize_configs(Int4WeightOnlyConfig(group_size = 128))[1]
    )
    out = [(q(e.t().contiguous()).t() if transposed else q(e.contiguous())) for e in w]
    return torch.stack(out).detach()


def _prepare(model, scheme = "int4"):
    from unsloth.models._utils import _prepare_model_for_qat
    return _prepare_model_for_qat(model, scheme)


@pytest.mark.parametrize("arch", ["gpt_oss", "qwen3_moe"])
def test_trainable_stacks_are_fake_quantized_every_forward(arch):
    model = _tiny_moe(arch)
    before = {k: v.detach().clone() for k, v in _stacks(model).items()}
    assert before, "no packed expert stacks"
    ids = torch.randint(0, 256, (2, 12), generator = torch.Generator().manual_seed(1))
    plain = _tiny_moe(arch)
    with torch.no_grad():
        want_plain = plain(input_ids = ids).logits
    model = _prepare(model)
    seen = []  # what the expert forward reads, captured after Unsloth's own pre-hooks
    owners = {
        n: m for n, m in model.named_modules() if m._parameters.get("gate_up_proj") is not None
    }
    for n, m in owners.items():
        m.register_forward_pre_hook(
            lambda mod, args, n = n: seen.append(
                (
                    f"{n}.gate_up_proj",
                    mod.gate_up_proj.detach().clone(),
                    f"{n}.down_proj",
                    mod.down_proj.detach().clone(),
                )
            )
        )
    out = model(input_ids = ids, labels = ids)
    assert len(seen) == len(owners)  # once per experts module per forward
    transposed = arch == "gpt_oss"
    for gname, gread, dname, dread in seen:
        for name, read in ((gname, gread), (dname, dread)):
            assert not torch.equal(read, before[name])
            assert torch.equal(read, _reference_int4(before[name], transposed)), name
    assert not torch.allclose(out.logits, want_plain)
    out.loss.backward()
    for name, p in _stacks(model).items():
        assert torch.equal(p.detach(), before[name])  # the 16-bit master weight is kept
        assert p.grad is not None and p.grad.abs().sum() > 0  # straight-through gradient
    from unsloth.precision import engaged

    linears = sum(1 for m in model.modules() if "FakeQuantized" in type(m).__name__)
    assert engaged(model)["qat:int4"]["modules"] == linears + len(before)


@pytest.mark.parametrize("arch", ["gpt_oss", "qwen3_moe"])
def test_frozen_stacks_are_rounded_once_and_match_the_per_forward_path(arch):
    ids = torch.randint(0, 256, (2, 12), generator = torch.Generator().manual_seed(1))
    trainable = _prepare(_tiny_moe(arch))
    frozen = _tiny_moe(arch)
    originals = {k: v.detach().clone() for k, v in _stacks(frozen).items()}
    for p in _stacks(frozen).values():
        p.requires_grad_(False)
    frozen = _prepare(frozen)
    for name, p in _stacks(frozen).items():
        transposed = arch == "gpt_oss"
        assert torch.equal(p.detach(), _reference_int4(originals[name], transposed)), name
    owners = [m for m in frozen.modules() if "_unsloth_qat_expert_hooks" in m.__dict__]
    assert not owners  # rounded: no per-forward fake quant left
    with torch.no_grad():
        a = trainable(input_ids = ids).logits
        b = frozen(input_ids = ids).logits
    assert torch.equal(a, b)


def test_kill_switch_and_unsupported_shape_warning(monkeypatch):
    monkeypatch.setenv("UNSLOTH_QAT_EXPERT_STACKS", "0")
    model = _tiny_moe("gpt_oss")
    before = {k: v.detach().clone() for k, v in _stacks(model).items()}
    for p in _stacks(model).values():
        p.requires_grad_(False)
    model = _prepare(model)
    assert all(
        torch.equal(p, before[k]) for k, p in _stacks(model).items()
    )  # the previous behaviour
    monkeypatch.delenv("UNSLOTH_QAT_EXPERT_STACKS")
    # intermediate 64: the transposed down_proj contracts over K = 64, which int4's group-size filter (128) refuses.
    model = _tiny_moe("gpt_oss", intermediate = 64)
    with pytest.warns(UserWarning, match = "unquantized.*down_proj \\(no_linear_recipe\\)"):
        _prepare(model)


def test_lora_on_the_expert_stacks_trains_on_the_rounded_base():
    """PEFT target_parameters LoRA on gate_up_proj / down_proj: the frozen stacks are rounded once and the adapter
    trains on top of them."""
    peft = pytest.importorskip("peft")
    model = _tiny_moe("gpt_oss")
    originals = {k: v.detach().clone() for k, v in _stacks(model).items()}
    cfg = peft.LoraConfig(
        r = 4, lora_alpha = 8, target_modules = ["q_proj"],
        target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )  # fmt: skip
    model = peft.get_peft_model(model, cfg)
    model = _prepare(model)
    stacks = {
        k.replace("base_model.model.", "").replace(".base_layer", ""): p
        for k, p in _stacks(model).items()
    }
    assert len(stacks) == len(originals)
    for name, p in stacks.items():
        assert not p.requires_grad
        assert torch.equal(p.detach(), _reference_int4(originals[name], True)), name
    ids = torch.randint(0, 256, (2, 12), generator = torch.Generator().manual_seed(1))
    model(input_ids = ids, labels = ids).loss.backward()
    expert_lora = [p for n, p in model.named_parameters() if "lora_B" in n and "experts" in n]
    assert expert_lora and all(p.grad is not None and p.grad.abs().sum() > 0 for p in expert_lora)


def test_mixed_recipes_take_the_linear_one():
    """int8-int4 lists an int8 embedding recipe before the int4 PerGroup(32) Linear recipe: experts get the latter."""
    from torchao.quantization import Int8DynamicActivationIntxWeightConfig
    from torchao.quantization.granularity import PerGroup
    from torchao.quantization.qat.fake_quantize_config import _infer_fake_quantize_configs
    from torchao.quantization.qat.fake_quantizer import FakeQuantizerBase

    model = _tiny_moe("gpt_oss")
    originals = {k: v.detach().clone() for k, v in _stacks(model).items()}
    for p in _stacks(model).values():
        p.requires_grad_(False)
    model = _prepare(model, "int8-int4")
    cfg = Int8DynamicActivationIntxWeightConfig(
        weight_dtype = torch.int4, weight_granularity = PerGroup(32)
    )
    q = FakeQuantizerBase.from_config(_infer_fake_quantize_configs(cfg)[1])
    for name, p in _stacks(model).items():
        want = torch.stack([q(e.t().contiguous()).t() for e in originals[name]]).detach()
        assert torch.equal(p.detach(), want), name
        assert (
            torch.unique(p.detach()[0, :32, 0]).numel() <= 16
        )  # one int4 group: at most 16 levels


def test_name_filters_are_checked_for_every_stack():
    """A custom TorchAOConfig whose filters pick recipes by name: same-shaped stacks follow their own names."""
    from torchao.quantization import Int4WeightOnlyConfig, IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerAxis

    from unsloth.models._utils import TorchAOConfig

    int8 = IntxWeightOnlyConfig(weight_dtype = torch.int8, granularity = PerAxis(0))
    config = TorchAOConfig(
        qat_scheme = None,
        base_config_and_filter_fns = [
            (Int4WeightOnlyConfig(group_size = 128), lambda m, fqn: isinstance(m, torch.nn.Linear) and "layers.0." in fqn),
            (int8, lambda m, fqn: isinstance(m, torch.nn.Linear) and "layers.1." in fqn),
        ],
    )  # fmt: skip
    model = _tiny_moe("gpt_oss")
    originals = {k: v.detach().clone() for k, v in _stacks(model).items()}
    for p in _stacks(model).values():
        p.requires_grad_(False)
    model = _prepare(model, config)
    from torchao.quantization.qat.fake_quantize_config import _infer_fake_quantize_configs
    from torchao.quantization.qat.fake_quantizer import FakeQuantizerBase

    q8 = FakeQuantizerBase.from_config(_infer_fake_quantize_configs(int8)[1])
    for name, p in _stacks(model).items():
        if "layers.0." in name:
            want = _reference_int4(originals[name], True)
        else:
            want = torch.stack([q8(e.t().contiguous()).t() for e in originals[name]]).detach()
        assert torch.equal(p.detach(), want), name


def test_frozen_stacks_stay_rounded_across_load_state_dict():
    model = _tiny_moe("gpt_oss")
    originals = {k: v.detach().clone() for k, v in _stacks(model).items()}
    unrounded = {k: v.clone() for k, v in model.state_dict().items()}
    for p in _stacks(model).values():
        p.requires_grad_(False)
    model = _prepare(model)
    rounded = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(unrounded)  # an unrounded checkpoint: rounded again
    for name, p in _stacks(model).items():
        assert torch.equal(p.detach(), _reference_int4(originals[name], True)), name
    model.load_state_dict(rounded)  # its own rounded values: kept, not quantized a second time
    for k, v in model.state_dict().items():
        assert torch.equal(v, rounded[k]), k
    model.load_state_dict(unrounded, assign = True)  # new Parameter objects
    for name, p in _stacks(model).items():
        assert torch.equal(p.detach(), _reference_int4(originals[name], True)), name


def test_rounded_checkpoint_from_another_prepared_model_is_not_quantized_twice():
    """int8 fake quant is not idempotent: a rounded stack quantized again moves. Loading another prepared model's
    state dict must reproduce its stacks exactly (its metadata marks them rounded)."""
    src = _tiny_moe("gpt_oss")
    for p in _stacks(src).values():
        p.requires_grad_(False)
    src = _prepare(src, "int8")
    torch.manual_seed(123)
    dst = _tiny_moe("gpt_oss")
    with torch.no_grad():
        for p in _stacks(dst).values():
            p.normal_()  # differently initialized
            p.requires_grad_(False)
    dst = _prepare(dst, "int8")
    sd = src.state_dict()
    dst.load_state_dict(sd)
    for name, p in _stacks(dst).items():
        assert torch.equal(p.detach(), _stacks(src)[name].detach()), name


def test_rounded_checkpoint_from_another_recipe_is_rounded_again():
    """Rounded provenance only counts for the same recipe: int8-rounded stacks loaded into an int4 model get int4."""
    src = _tiny_moe("gpt_oss")
    for p in _stacks(src).values():
        p.requires_grad_(False)
    src = _prepare(src, "int8")
    dst = _tiny_moe("gpt_oss")
    for p in _stacks(dst).values():
        p.requires_grad_(False)
    dst = _prepare(dst, "int4")
    dst.load_state_dict(src.state_dict())
    for name, p in _stacks(dst).items():
        want = _reference_int4(_stacks(src)[name].detach(), True)
        assert torch.equal(p.detach(), want), name
        assert not torch.equal(p.detach(), _stacks(src)[name].detach())
