# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A multi-LoRA expert export loads in vanilla PEFT with its trained delta, for experts Zoo re-lays out."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.kernels import multi_lora as ml

transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")
pytestmark = pytest.mark.skipif(
    not hasattr(transformers, "Qwen3VLMoeForConditionalGeneration"),
    reason = "transformers has no Qwen3-VL-MoE",
)

EXPERTS = ["experts.gate_up_proj", "experts.down_proj"]
R, ALPHA = 4, 8.0

# Shared by this process and the vanilla child, so both build the same architecture.
BUILD = textwrap.dedent(
    """
    import torch
    import transformers

    def build():
        text = dict(vocab_size = 256, hidden_size = 64, intermediate_size = 48, moe_intermediate_size = 40,
            num_hidden_layers = 1, num_attention_heads = 4, num_key_value_heads = 2, head_dim = 16,
            num_experts = 4, num_experts_per_tok = 2, max_position_embeddings = 128, use_cache = False,
            pad_token_id = 0)
        rope = {"rope_type": "default", "mrope_section": [2, 3, 3]}
        if int(transformers.__version__.split(".")[0]) >= 5:
            text["rope_parameters"] = dict(rope, rope_theta = 10000.0)
        else:
            text.update(rope_scaling = rope, rope_theta = 10000.0)
        vision = dict(depth = 1, hidden_size = 32, intermediate_size = 64, num_heads = 2, out_hidden_size = 64,
            deepstack_visual_indexes = [0], num_position_embeddings = 16)
        config = transformers.Qwen3VLMoeConfig(text_config = text, vision_config = vision,
                                               tie_word_embeddings = False)
        torch.manual_seed(0)
        return transformers.Qwen3VLMoeForConditionalGeneration._from_config(config).float().eval()
    """
)

VANILLA = BUILD + textwrap.dedent(
    """
    import sys
    import peft
    assert "unsloth" not in sys.modules and "unsloth_zoo" not in sys.modules
    adapter_dir, out = sys.argv[1], sys.argv[2]
    model = peft.PeftModel.from_pretrained(build(), adapter_dir)
    deltas = {}
    for name, module in model.named_modules():
        if type(module).__name__ != "ParamWrapper":
            continue
        name = name.removeprefix("base_model.model.")
        while name.endswith(".base_layer"):
            name = name[: -len(".base_layer")]
        deltas[f"{name}.{module.parameter_name}"] = module.get_delta_weight("default").detach().clone()
    torch.save(deltas, out)
    """
)


def _apply_zoo_patches():
    from unsloth_zoo.temporary_patches.common import TEMPORARY_PATCHES
    for patch in TEMPORARY_PATCHES:
        try:
            patch()
        except Exception:
            pass


def _key(name, parameter_name):
    name = name.removeprefix("base_model.model.")
    while name.endswith(".base_layer"):
        name = name[: -len(".base_layer")]
    return f"{name}.{parameter_name}"


def _export(tmp_path):
    """A tiny Qwen3-VL-MoE with adapter 0 trained (non-zero B) and exported as Studio's batched trainer does."""
    swap = pytest.importorskip("unsloth_zoo.temporary_patches.moe_lora_factor_swap")
    _apply_zoo_patches()
    scope = {}
    exec(BUILD, scope)
    model = scope["build"]()
    lm = model.model.language_model
    experts_cls = type(lm.layers[0].mlp.experts)
    if not swap._factors_swapped(experts_cls):
        pytest.skip(f"this Zoo does not re-lay out {experts_cls.__name__}")
    ml.inject_multi_lora(lm, ["q_proj"], 2, R, ALPHA)
    assert ml.inject_multi_lora_experts(lm, EXPERTS, 2, R, ALPHA)
    with torch.no_grad():
        # Non-zero B: PEFT's zero init would load any layout as a zero delta.
        torch.manual_seed(1)
        for _, layer in ml._layers(model):
            for B in layer.lora_B:
                B.normal_(0, 0.5)

    from safetensors.torch import save_file

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    state = {k: v.detach().cpu().contiguous() for k, v in ml.adapter_state_dict(model, 0).items()}
    save_file(state, str(adapter_dir / "adapter_model.safetensors"), metadata = {"format": "pt"})
    peft.LoraConfig(
        r = R,
        lora_alpha = ALPHA,
        lora_dropout = 0.0,
        target_modules = ["q_proj"],
        target_parameters = EXPERTS,
    ).save_pretrained(str(adapter_dir))
    return model, scope["build"], adapter_dir


def _vanilla_load(tmp_path, adapter_dir):
    script = tmp_path / "vanilla_load.py"
    script.write_text(VANILLA)
    out = tmp_path / "deltas.pt"
    env = {k: v for k, v in os.environ.items() if not k.startswith("UNSLOTH")}
    proc = subprocess.run(
        [sys.executable, "-I", str(script), str(adapter_dir), str(out)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 600,
    )
    return proc, out


def test_multi_lora_expert_export_loads_in_vanilla_peft(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_MOE_LORA_B_LAYOUT", raising = False)
    model, build, adapter_dir = _export(tmp_path)
    mine = {}
    for name, experts in model.named_modules():
        if isinstance(experts, ml.MultiLoraExperts):
            for param in experts.parameter_names:
                W = getattr(experts.base_layer, param)
                delta = experts.adapters[param].folded(torch.zeros_like(W), 0, experts.transposed)
                mine[f"{name}.{param.split('.')[-1]}"] = delta.detach()
    assert len(mine) == len(EXPERTS)

    proc, out = _vanilla_load(tmp_path, adapter_dir)
    assert proc.returncode == 0, proc.stderr[-3000:]
    vanilla = torch.load(out)
    assert set(vanilla) == set(mine), (sorted(vanilla), sorted(mine))
    for key, delta in mine.items():
        # Qwen3-VL-MoE: Zoo stores [E, H, 2I] / [E, I, H] where vanilla stores [E, 2I, H] / [E, H, I].
        assert vanilla[key].shape == delta.transpose(1, 2).shape, key
        assert vanilla[key].abs().max() > 1e-3, key
        torch.testing.assert_close(vanilla[key], delta.transpose(1, 2), atol = 1e-6, rtol = 1e-5)

    # The same folder still reloads under Unsloth (Zoo maps the vanilla pair back) with the trained delta.
    reloaded = peft.PeftModel.from_pretrained(build(), str(adapter_dir), torch_device = "cpu")
    seen = 0
    for name, module in reloaded.named_modules():
        if type(module).__name__ == "ParamWrapper":
            key = _key(name, module.parameter_name)
            torch.testing.assert_close(
                module.get_delta_weight("default"), mine[key], atol = 1e-6, rtol = 1e-5
            )
            seen += 1
    assert seen == len(mine)


def test_legacy_grouped_multi_lora_export_is_refused_by_vanilla_peft(tmp_path, monkeypatch):
    """A legacy grouped lora_B saved 2D would load with experts mixed, so it is saved [out, E, R] and refused."""
    swap = pytest.importorskip("unsloth_zoo.temporary_patches.moe_lora_factor_swap")
    if not hasattr(swap, "_flatten_grouped_lora_B"):
        pytest.skip("this Zoo saves a legacy grouped_by_expert lora_B 2D")
    monkeypatch.setenv("UNSLOTH_MOE_LORA_B_LAYOUT", "grouped_by_expert")
    model, build, adapter_dir = _export(tmp_path)
    trained = {}
    for name, experts in model.named_modules():
        if isinstance(experts, ml.MultiLoraExperts):
            for param in experts.parameter_names:
                A, B = experts.adapters[param].peft_tensors(0, "grouped_by_expert")
                trained[f"{name}.{param.split('.')[-1]}"] = (A, B)

    from safetensors.torch import load_file

    saved = load_file(str(adapter_dir / "adapter_model.safetensors"))
    expert_B = [v for k, v in saved.items() if "experts" in k and ".lora_B." in k]
    assert len(expert_B) == len(EXPERTS)
    # A 3D lora_B is refused by shape on every PEFT, including 0.18 where a 2D one would load mixed.
    assert all(B.dim() == 3 and B.shape[1] == 4 for B in expert_B), [
        tuple(B.shape) for B in expert_B
    ]
    proc, _ = _vanilla_load(tmp_path, adapter_dir)
    assert proc.returncode != 0 and "size mismatch" in proc.stderr, proc.stderr[-3000:]

    reloaded = peft.PeftModel.from_pretrained(build(), str(adapter_dir), torch_device = "cpu")
    seen = 0
    for name, module in reloaded.named_modules():
        if type(module).__name__ == "ParamWrapper":
            A, B = trained[_key(name, module.parameter_name)]
            torch.testing.assert_close(module.lora_A["default"].weight, A, atol = 0, rtol = 0)
            torch.testing.assert_close(module.lora_B["default"].weight, B, atol = 0, rtol = 0)
            seen += 1
    assert seen == len(trained)
