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
            self.vision_tower.vision_model.encoder.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.vision_tower.vision_model.encoder.layers[0].self_attn = torch.nn.Module()
            self.vision_tower.vision_model.encoder.layers[0].self_attn.q_proj = torch.nn.Linear(
                4, 4
            )

    regex = get_peft_regex(
        FakeVLM(),
        finetune_vision_layers = False,
        finetune_language_layers = True,
        finetune_attention_modules = True,
        finetune_mlp_modules = True,
        target_modules = ["q_proj"],
    )

    assert re.search(regex, "language_model.layers.0.self_attn.q_proj")
    assert not re.search(regex, "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj")


def test_fast_vision_model_wraps_explicit_targets_when_layer_filters_are_used():
    source = Path("unsloth/models/vision.py").read_text(encoding = "utf-8")

    assert "target_modules = get_peft_regex(" in source
    assert "target_modules = list(target_modules)" in source


def test_embedding_redirect_respects_disabled_language_layers():
    from unsloth.models._utils import _redirect_embedding_targets

    targets = ["q_proj", "embed_tokens", "lm_head"]
    adjusted, modules_to_save, moved, direct_target = _redirect_embedding_targets(
        targets,
        None,
        allow_redirect = False,
    )

    assert adjusted is targets
    assert modules_to_save is None
    assert moved == ()
    assert direct_target is False


def test_embedding_redirect_moves_embeddings_when_lora_targets_remain():
    from unsloth.models._utils import _redirect_embedding_targets

    adjusted, modules_to_save, moved, direct_target = _redirect_embedding_targets(
        ["q_proj", "embed_tokens", "lm_head"],
        ["embed_tokens"],
    )

    assert adjusted == ["q_proj"]
    assert modules_to_save == ["embed_tokens", "lm_head"]
    assert moved == ("embed_tokens", "lm_head")
    assert direct_target is False


def test_embedding_redirect_keeps_legacy_lm_head_only_target_valid():
    from unsloth.models._utils import _redirect_embedding_targets

    adjusted, modules_to_save, moved, direct_target = _redirect_embedding_targets(
        ["lm_head"],
        None,
        preserve_lm_head_target = True,
    )

    assert adjusted == ["lm_head"]
    assert modules_to_save is None
    assert moved == ()
    assert direct_target is True


def test_embedding_redirect_keeps_a_lora_target_with_both_embeddings():
    from unsloth.models._utils import _redirect_embedding_targets

    adjusted, modules_to_save, moved, direct_target = _redirect_embedding_targets(
        ["embed_tokens", "lm_head"],
        None,
        preserve_lm_head_target = True,
    )

    assert adjusted == ["lm_head"]
    assert modules_to_save == ["embed_tokens"]
    assert moved == ("embed_tokens",)
    assert direct_target is True


def test_embedding_redirect_keeps_embed_tokens_when_it_is_the_only_target():
    from unsloth.models._utils import _redirect_embedding_targets

    adjusted, modules_to_save, moved, direct_target = _redirect_embedding_targets(
        ["embed_tokens"],
        None,
        preserve_embedding_target = True,
    )

    assert adjusted == ["embed_tokens"]
    assert modules_to_save is None
    assert moved == ()
    assert direct_target is True


def test_embedding_redirect_keeps_lm_head_lora_for_fast_inference():
    from unsloth.models._utils import _redirect_embedding_targets

    adjusted, modules_to_save, moved, direct_target = _redirect_embedding_targets(
        ["q_proj", "lm_head"],
        None,
        preserve_lm_head_target = True,
        redirect_lm_head = False,
    )

    assert adjusted == ["q_proj", "lm_head"]
    assert modules_to_save is None
    assert moved == ()
    assert direct_target is False


def test_embedding_redirect_deduplicates_preserved_target():
    from unsloth.models._utils import _redirect_embedding_targets

    adjusted, modules_to_save, moved, direct_target = _redirect_embedding_targets(
        ["lm_head", "lm_head"],
        None,
        preserve_lm_head_target = True,
    )

    assert adjusted == ["lm_head"]
    assert modules_to_save is None
    assert moved == ()
    assert direct_target is True
