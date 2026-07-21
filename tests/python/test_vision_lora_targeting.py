from pathlib import Path
import re

import torch


def test_vlm_lora_regex_respects_language_only_with_explicit_targets():
    from unsloth_zoo.peft_utils import get_peft_regex

    class FakeVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = torch.nn.Module()
            self.language_model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.language_model.layers[0].self_attn = torch.nn.Module()
            self.language_model.layers[0].self_attn.q_proj = torch.nn.Linear(4, 4)
            self.vision_tower = torch.nn.Module()
            self.vision_tower.vision_model = torch.nn.Module()
            self.vision_tower.vision_model.encoder = torch.nn.Module()
            self.vision_tower.vision_model.encoder.layers = torch.nn.ModuleList(
                [torch.nn.Module()]
            )
            self.vision_tower.vision_model.encoder.layers[
                0
            ].self_attn = torch.nn.Module()
            self.vision_tower.vision_model.encoder.layers[
                0
            ].self_attn.q_proj = torch.nn.Linear(4, 4)

    regex = get_peft_regex(
        FakeVLM(),
        finetune_vision_layers = False,
        finetune_language_layers = True,
        finetune_attention_modules = True,
        finetune_mlp_modules = True,
        target_modules = ["q_proj"],
    )

    assert re.search(regex, "language_model.layers.0.self_attn.q_proj")
    assert not re.search(
        regex, "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj"
    )


def test_fast_vision_model_wraps_explicit_targets_when_layer_filters_are_used():
    source = Path("unsloth/models/vision.py").read_text()

    assert "target_modules = get_peft_regex(" in source
    assert "target_modules = list(target_modules)" in source
