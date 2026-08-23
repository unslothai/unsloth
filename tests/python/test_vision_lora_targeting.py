from pathlib import Path
import re

import pytest
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
    adjusted, modules_to_save, moved = _redirect_embedding_targets(
        targets,
        None,
        allow_redirect = False,
    )

    assert adjusted is targets
    assert modules_to_save is None
    assert moved == ()


def test_embedding_redirect_moves_embeddings_when_lora_targets_remain():
    from unsloth.models._utils import _redirect_embedding_targets

    adjusted, modules_to_save, moved = _redirect_embedding_targets(
        ["q_proj", "embed_tokens", "lm_head"],
        ["embed_tokens"],
    )

    assert adjusted == ["q_proj"]
    assert modules_to_save == ["embed_tokens", "lm_head"]
    assert moved == ("embed_tokens", "lm_head")


def test_embedding_redirect_always_moves_lm_head():
    """LoRA on lm_head never trains, so it is redirected even when listed alone."""
    from unsloth.models._utils import _redirect_embedding_targets

    adjusted, modules_to_save, moved = _redirect_embedding_targets(["lm_head"], None)

    assert adjusted == []
    assert modules_to_save == ["lm_head"]
    assert moved == ("lm_head",)


def test_embedding_redirect_raises_when_no_lora_target_remains():
    from unsloth.models._utils import (
        _raise_if_no_lora_targets_left,
        _redirect_embedding_targets,
    )

    adjusted, modules_to_save, moved = _redirect_embedding_targets(
        ["embed_tokens", "lm_head"],
        None,
    )

    assert adjusted == []
    assert modules_to_save == ["embed_tokens", "lm_head"]
    with pytest.raises(RuntimeError, match = "target_modules` is now empty"):
        _raise_if_no_lora_targets_left(adjusted, moved)


def test_embedding_redirect_keeps_valid_target_lists_unchanged():
    from unsloth.models._utils import (
        _raise_if_no_lora_targets_left,
        _redirect_embedding_targets,
    )

    adjusted, modules_to_save, moved = _redirect_embedding_targets(["q_proj"], None)

    assert adjusted == ["q_proj"]
    assert modules_to_save is None
    assert moved == ()
    _raise_if_no_lora_targets_left(adjusted, moved)


def test_embedding_redirect_keeps_string_modules_to_save_whole():
    """list() on a str would shred it into single characters."""
    from unsloth.models._utils import _redirect_embedding_targets

    adjusted, modules_to_save, moved = _redirect_embedding_targets(
        ["q_proj", "embed_tokens"],
        "lm_head",
    )

    assert adjusted == ["q_proj"]
    assert modules_to_save == ["lm_head", "embed_tokens"]
    assert moved == ("embed_tokens",)


def test_embedding_redirect_rejects_embeddings_for_fast_inference():
    from unsloth.models._utils import (
        _raise_if_fast_inference_modules_to_save,
        _redirect_embedding_targets,
    )

    class FastInferenceModel:
        vllm_engine = object()

    adjusted, modules_to_save, moved = _redirect_embedding_targets(
        ["q_proj", "embed_tokens", "lm_head"],
        None,
    )

    assert adjusted == ["q_proj"]
    assert modules_to_save == ["embed_tokens", "lm_head"]
    assert moved == ("embed_tokens", "lm_head")
    with pytest.raises(NotImplementedError, match = "embed_tokens, lm_head"):
        _raise_if_fast_inference_modules_to_save(FastInferenceModel(), modules_to_save)


def test_fast_inference_guard_ignores_empty_and_inactive_engines():
    from unsloth.models._utils import _raise_if_fast_inference_modules_to_save

    class FastInferenceModel:
        vllm_engine = object()

    class InactiveModel:
        vllm_engine = None

    # An empty list names no trainable module, and vllm_engine = None is not fast inference.
    _raise_if_fast_inference_modules_to_save(FastInferenceModel(), [])
    _raise_if_fast_inference_modules_to_save(FastInferenceModel(), None)
    _raise_if_fast_inference_modules_to_save(InactiveModel(), ["embed_tokens"])


def test_embedding_redirect_deduplicates_targets():
    from unsloth.models._utils import _redirect_embedding_targets

    adjusted, modules_to_save, moved = _redirect_embedding_targets(
        ["q_proj", "q_proj", "lm_head", "lm_head"],
        None,
    )

    assert adjusted == ["q_proj"]
    assert modules_to_save == ["lm_head"]
    assert moved == ("lm_head",)


class _Cfg:
    def __init__(self, tie):
        self.tie_word_embeddings = tie


class _Model:
    def __init__(self, tie):
        self.config = _Cfg(tie)


def test_ensure_weight_tying_defaults_on_for_a_tied_pair_in_modules_to_save():
    """PEFT copies each saved module, so a tied pair diverges and merges wrong."""
    from unsloth.models._utils import _resolve_ensure_weight_tying

    both = ["embed_tokens", "lm_head"]
    assert _resolve_ensure_weight_tying(_Model(tie = True), both, None) is True
    assert _resolve_ensure_weight_tying(_Model(tie = False), both, None) is False


def test_ensure_weight_tying_covers_callers_that_pass_modules_to_save_themselves():
    """Studio's CPT path routes both names itself, so nothing is 'moved' by the redirect."""
    from unsloth.models._utils import _redirect_embedding_targets, _resolve_ensure_weight_tying

    target_modules, modules_to_save, moved = _redirect_embedding_targets(
        ["q_proj"],
        ["embed_tokens", "lm_head"],
    )
    assert moved == ()
    assert _resolve_ensure_weight_tying(_Model(tie = True), modules_to_save, None) is True


@pytest.mark.parametrize("modules_to_save", [None, [], ["embed_tokens"], ["lm_head"], ["score"]])
def test_ensure_weight_tying_stays_off_without_both_names(modules_to_save):
    from unsloth.models._utils import _resolve_ensure_weight_tying
    assert _resolve_ensure_weight_tying(_Model(tie = True), modules_to_save, None) is False


@pytest.mark.parametrize("requested", [True, False])
def test_explicit_ensure_weight_tying_always_wins(requested):
    from unsloth.models._utils import _resolve_ensure_weight_tying

    both = ["embed_tokens", "lm_head"]
    assert _resolve_ensure_weight_tying(_Model(tie = True), both, requested) is requested
    assert _resolve_ensure_weight_tying(_Model(tie = False), both, requested) is requested


def test_ensure_weight_tying_handles_models_without_a_config():
    from unsloth.models._utils import _resolve_ensure_weight_tying
    class Bare:
        pass

    assert _resolve_ensure_weight_tying(Bare(), ["embed_tokens", "lm_head"], None) is False


def test_target_parameters_alone_is_a_valid_lora_target():
    """PEFT accepts target_parameters with no target_modules (lora/model.py _prepare_adapter_config)."""
    from unsloth.models._utils import _raise_if_no_lora_targets_left
    _raise_if_no_lora_targets_left([], ("embed_tokens", "lm_head"), ["experts.gate_up_proj"])


@pytest.mark.parametrize("target_parameters", [None, []])
def test_empty_target_parameters_still_raises(target_parameters):
    from unsloth.models._utils import _raise_if_no_lora_targets_left
    with pytest.raises(RuntimeError, match = "target_modules` is now empty"):
        _raise_if_no_lora_targets_left([], ("embed_tokens",), target_parameters)
