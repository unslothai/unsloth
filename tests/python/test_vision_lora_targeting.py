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
    """Unsloth's CPT path routes both names itself, so nothing is 'moved' by the redirect."""
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


def test_tying_leaves_the_output_module_for_peft_to_reconstruct():
    """peft 0.18 ties only what modules_to_save does not name, so naming both there ties
    nothing and trains two copies that diverge."""
    from unsloth.models._utils import _drop_tied_output_module

    tied, untied = _Model(tie = True), _Model(tie = False)
    both = ["embed_tokens", "lm_head"]
    assert _drop_tied_output_module(tied, both, True) == ["embed_tokens"]
    assert _drop_tied_output_module(tied, both, False) == both
    # An untied model has no counterpart for PEFT to rebuild, so dropping lm_head there would leave the head the caller
    # asked to train frozen.
    assert _drop_tied_output_module(untied, both, True) == both
    # Only a real pair is split: tying can be requested with no pair to tie, and dropping
    # the lone head would train nothing (or crash on None).
    assert _drop_tied_output_module(tied, ["embed_tokens", "score"], True) == [
        "embed_tokens",
        "score",
    ]
    assert _drop_tied_output_module(tied, ["lm_head"], True) == ["lm_head"]
    assert _drop_tied_output_module(tied, ["lm_head"], False) == ["lm_head"]
    assert _drop_tied_output_module(tied, None, True) is None
    assert _drop_tied_output_module(tied, [], True) == []


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


CORE = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class _VllmModel:
    """A model loaded with fast_inference = True carries a live engine."""

    vllm_engine = object()


def test_fast_inference_leaves_lm_head_alone():
    """Redirecting it would newly raise on GRPO scripts that run today: the pre-existing
    guard refuses any modules_to_save, and LoRA on lm_head never trained here anyway."""
    from unsloth.models._utils import (
        _redirect_embedding_targets,
        _vllm_unmovable_embedding_modules,
    )

    skip = _vllm_unmovable_embedding_modules(_VllmModel(), CORE + ["lm_head"])
    assert skip == ("lm_head",)

    targets, saved, moved = _redirect_embedding_targets(CORE + ["lm_head"], None, skip = skip)
    assert targets == CORE + ["lm_head"], "lm_head must stay a LoRA target under vLLM"
    assert saved is None, "an empty modules_to_save is what keeps the guard quiet"
    assert moved == ()

    # embed_tokens still redirects, so fast inference still refuses it exactly as before.
    targets, saved, moved = _redirect_embedding_targets(
        CORE + ["embed_tokens", "lm_head"],
        None,
        skip = skip,
    )
    assert saved == ["embed_tokens"] and moved == ("embed_tokens",)
    assert targets == CORE + ["lm_head"], "the spared name is kept, not silently dropped"


def test_without_fast_inference_nothing_is_skipped():
    from unsloth.models._utils import _vllm_unmovable_embedding_modules

    class Plain:
        pass

    assert _vllm_unmovable_embedding_modules(Plain(), CORE + ["lm_head"]) == ()
    # A regex/None target list must not crash the lm_head membership test.
    assert _vllm_unmovable_embedding_modules(_VllmModel(), None) == ("lm_head",)
    assert _vllm_unmovable_embedding_modules(_VllmModel(), "all-linear") == ("lm_head",)


def test_qualified_embedding_names_are_redirected_too():
    """PEFT resolves model.embed_tokens to the same module, so it must not be left in
    target_modules where LoRA on it never trains."""
    from unsloth.models._utils import (
        _drop_tied_output_module,
        _redirect_embedding_targets,
        _resolve_ensure_weight_tying,
    )

    targets, saved, moved = _redirect_embedding_targets(
        CORE + ["model.embed_tokens", "language_model.lm_head"],
        None,
    )
    assert targets == CORE
    # The caller's spelling is preserved: PEFT matches modules_to_save on the suffix.
    assert saved == ["model.embed_tokens", "language_model.lm_head"]
    assert moved == ("model.embed_tokens", "language_model.lm_head")

    # A qualified pair is still recognised as a tied pair.
    assert _resolve_ensure_weight_tying(_Model(tie = True), saved, None) is True
    # Normalized to the bare name so peft 0.18 recognises it as the embedding to retie.
    assert _drop_tied_output_module(_Model(tie = True), saved, True) == ["embed_tokens"]


def test_only_embedding_names_are_matched_by_leaf():
    """layers.0.q_proj is a real target and must never be treated as an embedding."""
    from unsloth.models._utils import _embedding_leaf, _redirect_embedding_targets

    assert _embedding_leaf("model.embed_tokens") == "embed_tokens"
    assert _embedding_leaf("lm_head") == "lm_head"
    assert _embedding_leaf("layers.0.q_proj") is None
    assert _embedding_leaf("q_proj") is None
    assert _embedding_leaf(None) is None

    targets, saved, moved = _redirect_embedding_targets(["layers.0.q_proj"], None)
    assert targets == ["layers.0.q_proj"] and saved is None and moved == ()


def test_tying_normalizes_the_saved_embedding_to_its_bare_name():
    """peft 0.18 ties `tied_weight_keys` minus whole modules_to_save entries, so a
    qualified model.embed_tokens matches nothing and no output module is rebuilt."""
    from unsloth.models._utils import _drop_tied_output_module

    tied = _Model(tie = True)
    assert _drop_tied_output_module(
        tied,
        ["model.embed_tokens", "language_model.lm_head"],
        True,
    ) == ["embed_tokens"]
    assert _drop_tied_output_module(tied, ["embed_tokens", "lm_head"], True) == ["embed_tokens"]
    # Untouched when tying does not apply.
    assert _drop_tied_output_module(
        tied,
        ["model.embed_tokens", "lm_head"],
        False,
    ) == ["model.embed_tokens", "lm_head"]
    # Non-embedding names keep their full path.
    assert _drop_tied_output_module(
        tied,
        ["model.embed_tokens", "lm_head", "custom.score"],
        True,
    ) == ["embed_tokens", "custom.score"]


def test_tying_needs_the_input_embedding_saved():
    """PEFT rebuilds the output from the input embedding's wrapper, so saving only
    lm_head cannot tie: peft 0.18 raises AttributeError, later versions tie nothing."""
    from unsloth.models._utils import _effective_weight_tying

    tied, untied = _Model(tie = True), _Model(tie = False)
    assert _effective_weight_tying(tied, ["embed_tokens", "lm_head"], None) is True
    assert _effective_weight_tying(tied, ["embed_tokens"], True) is True
    assert _effective_weight_tying(tied, ["model.embed_tokens"], True) is True
    assert _effective_weight_tying(tied, ["lm_head"], True) is False
    assert _effective_weight_tying(tied, None, True) is False
    assert _effective_weight_tying(tied, ["score"], True) is False
    assert _effective_weight_tying(untied, ["embed_tokens", "lm_head"], True) is False
    # An explicit False always wins.
    assert _effective_weight_tying(tied, ["embed_tokens", "lm_head"], False) is False


class _Composite(torch.nn.Module):
    """Audio tower and language model both expose an embed_tokens leaf."""

    def __init__(self, two_embeddings):
        super().__init__()
        self.language_model = torch.nn.Module()
        self.language_model.embed_tokens = torch.nn.Embedding(8, 4)
        if two_embeddings:
            self.audio_tower = torch.nn.Module()
            self.audio_tower.embed_tokens = torch.nn.Embedding(8, 4)
        self.config = _Cfg(True)


def test_a_qualified_name_is_not_widened_on_a_composite_model():
    """PEFT suffix-matches, so the bare leaf would wrap the audio tower's embedding too."""
    from unsloth.models._utils import _drop_tied_output_module

    pair = ["language_model.embed_tokens", "lm_head"]
    # Two candidates: keep the caller's scope, and keep the head trainable with it.
    assert _drop_tied_output_module(_Composite(True), list(pair), True) == pair
    # One candidate: rewriting is safe and peft 0.18 needs the bare name to retie.
    assert _drop_tied_output_module(_Composite(False), list(pair), True) == ["embed_tokens"]
