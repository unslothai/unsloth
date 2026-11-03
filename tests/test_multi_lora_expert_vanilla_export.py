# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A multi-LoRA expert export (adapter_state_dict, what Studio's batched trainer saves) loads in vanilla
transformers + PEFT with the delta it trained, for experts Zoo re-lays out (Qwen3-VL-MoE stores its stacks
transposed, so the installed PEFT factors them on the other axis than vanilla does)."""

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


def test_multi_lora_expert_export_loads_in_vanilla_peft(tmp_path):
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
    wrapped = ml.inject_multi_lora_experts(lm, EXPERTS, 2, R, ALPHA)
    assert wrapped
    with torch.no_grad():
        # Non-zero B: PEFT's zero init would load any layout as a zero delta.
        torch.manual_seed(1)
        for _, layer in ml._layers(model):
            for B in layer.lora_B:
                B.normal_(0, 0.5)

    # The delta adapter 0 trained, in Zoo's stored stack layout.
    mine = {}
    for name, experts in model.named_modules():
        if isinstance(experts, ml.MultiLoraExperts):
            for param in experts.parameter_names:
                W = getattr(experts.base_layer, param)
                delta = experts.adapters[param].folded(torch.zeros_like(W), 0, experts.transposed)
                mine[f"{name}.{param.split('.')[-1]}"] = delta.detach()
    assert len(mine) == len(EXPERTS)

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
    assert proc.returncode == 0, proc.stderr[-3000:]

    vanilla = torch.load(out)
    assert set(vanilla) == set(mine), (sorted(vanilla), sorted(mine))
    for key, delta in mine.items():
        # Qwen3-VL-MoE: Zoo stores [E, H, 2I] / [E, I, H] where vanilla stores [E, 2I, H] / [E, H, I].
        assert vanilla[key].shape == delta.transpose(1, 2).shape, key
        assert vanilla[key].abs().max() > 1e-3, key
        torch.testing.assert_close(vanilla[key], delta.transpose(1, 2), atol = 1e-6, rtol = 1e-5)

    # The same folder still reloads under Unsloth (Zoo maps the vanilla pair back) with the trained delta.
    reloaded = peft.PeftModel.from_pretrained(scope["build"](), str(adapter_dir))
    seen = 0
    for name, module in reloaded.named_modules():
        if type(module).__name__ != "ParamWrapper":
            continue
        name = name.removeprefix("base_model.model.")
        while name.endswith(".base_layer"):
            name = name[: -len(".base_layer")]
        key = f"{name}.{module.parameter_name}"
        torch.testing.assert_close(
            module.get_delta_weight("default"), mine[key], atol = 1e-6, rtol = 1e-5
        )
        seen += 1
    assert seen == len(mine)
