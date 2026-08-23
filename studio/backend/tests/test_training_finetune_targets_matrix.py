# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Full [training type] x [model branch] x [worker] x [selector combination] sweep.

The four finetune_* selectors are read by exactly two branches on CUDA (vision VLM and audio
VLM) and by every LoRA branch on MLX, and every other branch builds its adapter from
target_modules alone. This file pins that map so a guard cannot start firing on a branch that
never read the selectors, which would turn a previously working run into a hard error.
"""

import ast
import inspect
import itertools
import textwrap

import pytest

from core.training.worker import (
    _check_finetune_targets_after_detect,
    _check_mlx_finetune_targets,
    _check_mlx_effective_targets,
    _names_a_cpt_target,
    _finetune_selectors,
    _pre_detect_training_model,
    _requests_all_linear,
    _run_mlx_training,
)
from models import TrainingStartRequest


TRAINING_TYPES = ("LoRA/QLoRA", "Full Finetuning", "Continued Pretraining")

# The branch pre_detect settles on. Only "vlm" and "audio_vlm" forward the selectors on CUDA;
# prepare_model_for_training's other arms pass target_modules and never the four.
BRANCHES = ("text", "vlm", "audio_vlm", "codec", "whisper", "snac")
_CUDA_BRANCHES_READING_SELECTORS = ("vlm", "audio_vlm")

SELECTOR_CASES = {
    "omitted": {},
    "all_false": {
        "finetune_vision_layers": False,
        "finetune_language_layers": False,
        "finetune_attention_modules": False,
        "finetune_mlp_modules": False,
    },
    "all_true": {
        "finetune_vision_layers": True,
        "finetune_language_layers": True,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
    },
    "mlp_only": {
        "finetune_vision_layers": False,
        "finetune_language_layers": False,
        "finetune_attention_modules": False,
        "finetune_mlp_modules": True,
    },
    "vision_only": {
        "finetune_vision_layers": True,
        "finetune_language_layers": False,
        "finetune_attention_modules": False,
        "finetune_mlp_modules": False,
    },
    "vision_and_attention": {
        "finetune_vision_layers": True,
        "finetune_language_layers": False,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": False,
    },
    "language_and_mlp": {
        "finetune_vision_layers": False,
        "finetune_language_layers": True,
        "finetune_attention_modules": False,
        "finetune_mlp_modules": True,
    },
}


_DEFAULT_LEAVES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# target_modules changes which guard applies at all: "all-linear" turns every selector on
# inside get_peft_model, and MLX's filter only strips names it recognises as attention or MLP.
TARGET_MODULE_CASES = {
    "unset": None,
    "empty": [],
    "all_linear": ["all-linear"],
    "all_linear_plus_lm_head": ["all-linear", "lm_head"],
    "lm_head": ["lm_head"],
    "embed_tokens": ["embed_tokens"],
    "fused_qkv": ["Wqkv"],
    "architecture_specific": ["c_fc"],
    "defaults": list(_DEFAULT_LEAVES),
}


class _Trainer:
    """Stands in for the detection result pre_detect leaves on the trainer."""

    def __init__(self, branch: str):
        self.is_vlm = branch == "vlm"
        self.is_audio_vlm = branch == "audio_vlm"
        self._audio_type = {
            "codec": "csm",
            "whisper": "whisper",
            "snac": "snac",
        }.get(branch)


def _request_config(
    training_type: str,
    branch: str,
    selectors: dict,
    target_modules = None,
) -> dict:
    """Build the worker config the way /training/start does: through the request model, so
    an omitted field arrives as the request model's default rather than as a missing key."""
    request = TrainingStartRequest(
        model_name = "unsloth/Llama-3.2-1B-Instruct",
        training_type = training_type,
        format_type = "alpaca",
        target_modules = target_modules,
        **selectors,
    )
    # The route sends an empty list through as None, so the worker never sees a falsy list.
    config = {
        "training_type": training_type,
        "target_modules": request.target_modules if request.target_modules else None,
    }
    for field in (
        "finetune_vision_layers",
        "finetune_language_layers",
        "finetune_attention_modules",
        "finetune_mlp_modules",
    ):
        config[field] = getattr(request, field)
    if branch == "vlm":
        config["is_dataset_image"] = True
    if branch in ("audio_vlm", "codec", "whisper", "snac"):
        config["is_dataset_audio"] = True
    return config


def _cuda_guard_fires(config: dict, branch: str) -> bool:
    try:
        _check_finetune_targets_after_detect(_Trainer(branch), config)
    except ValueError:
        return True
    return False


def _mlx_guard_fires(config: dict) -> bool:
    if config.get("training_type", "LoRA/QLoRA") != "LoRA/QLoRA":
        return False  # the call site at _run_mlx_training sits under `if use_lora`
    try:
        _check_mlx_finetune_targets(config)
    except ValueError:
        return True
    return False


def _cuda_targets(config: dict, branch: str) -> str:
    """The adapter target set the CUDA worker ends up with, once the guard has passed."""
    training_type = config["training_type"]
    if training_type == "Full Finetuning":
        return "no adapter (full finetuning)"
    if training_type == "Continued Pretraining":
        return "target_modules only (q,k,v,o,gate,up,down,lm_head)"
    if branch not in _CUDA_BRANCHES_READING_SELECTORS:
        return "target_modules only (selectors ignored)"
    if _requests_all_linear(config):
        return "every linear layer (all-linear forces the selectors on)"
    vision, language, attention, mlp = _finetune_selectors(config)
    families = [name for name, on in (("vision", vision), ("language", language)) if on]
    modules = [name for name, on in (("attention", attention), ("mlp", mlp)) if on]
    return f"regex over {'+'.join(families)} x {'+'.join(modules)}"


def _mlx_targets(config: dict) -> str:
    training_type = config["training_type"]
    if training_type != "LoRA/QLoRA":
        return "no adapter (MLX applies LoRA only for LoRA/QLoRA)"
    is_vlm = bool(config.get("is_dataset_image", False))
    _, language, attention, mlp = _finetune_selectors(config)
    explicit = config.get("target_modules")
    if explicit and not (attention or mlp):
        # The filter drops only recognised attention and MLP leaves; whatever is left trains.
        return f"whatever survives the filter of {list(explicit)}"
    vision = bool(config.get("finetune_vision_layers", False)) if is_vlm else False
    if (attention or mlp) and not language and not vision:
        language = True  # the back-fill at the MLX LoRA branch
    families = [name for name, on in (("vision", vision), ("language", language)) if on]
    modules = [name for name, on in (("attention", attention), ("mlp", mlp)) if on]
    return f"{'+'.join(families)} x {'+'.join(modules)}"


@pytest.mark.parametrize("training_type", TRAINING_TYPES)
@pytest.mark.parametrize("branch", BRANCHES)
@pytest.mark.parametrize("selector_case", sorted(SELECTOR_CASES))
@pytest.mark.parametrize("targets_case", sorted(TARGET_MODULE_CASES))
def test_cuda_guard_only_fires_where_the_selectors_are_read(
    training_type, branch, selector_case, targets_case
):
    config = _request_config(
        training_type,
        branch,
        SELECTOR_CASES[selector_case],
        TARGET_MODULE_CASES[targets_case],
    )
    vision, language, attention, mlp = _finetune_selectors(config)

    expected = (
        training_type == "LoRA/QLoRA"
        and branch in _CUDA_BRANCHES_READING_SELECTORS
        and not _requests_all_linear(config)
        and (not (vision or language) or not (attention or mlp))
    )

    assert _cuda_guard_fires(config, branch) is expected
    if not expected:
        assert _cuda_targets(config, branch)


@pytest.mark.parametrize("training_type", TRAINING_TYPES)
@pytest.mark.parametrize("branch", BRANCHES)
@pytest.mark.parametrize("selector_case", sorted(SELECTOR_CASES))
@pytest.mark.parametrize("targets_case", sorted(TARGET_MODULE_CASES))
def test_mlx_guard_only_fires_on_an_empty_module_selection(
    training_type, branch, selector_case, targets_case
):
    config = _request_config(
        training_type,
        branch,
        SELECTOR_CASES[selector_case],
        TARGET_MODULE_CASES[targets_case],
    )
    vision, language, attention, mlp = _finetune_selectors(config)
    targets = config.get("target_modules")

    # Two rules, because the loader has two. With no explicit list the default seven are
    # wholly attention and MLP, so an empty module selection leaves nothing. With one, the
    # text branch also needs a layer family, and only a CPT target trains without one.
    if not targets:
        empty = not (attention or mlp)
    else:
        empty = not _names_a_cpt_target(targets) and not (attention or mlp or language or vision)
    expected = training_type == "LoRA/QLoRA" and empty

    assert _mlx_guard_fires(config) is expected


@pytest.mark.parametrize("branch", BRANCHES)
def test_omitted_selectors_never_trip_either_guard(branch):
    """The headline of the default flip: a caller that sends none of the four now trains
    the language attention and MLP modules on every branch instead of failing."""
    for training_type in TRAINING_TYPES:
        config = _request_config(training_type, branch, {})

        assert _cuda_guard_fires(config, branch) is False
        assert _mlx_guard_fires(config) is False

    lora = _request_config("LoRA/QLoRA", branch, {})
    if branch in _CUDA_BRANCHES_READING_SELECTORS:
        assert _cuda_targets(lora, branch) == "regex over language x attention+mlp"
    else:
        assert _cuda_targets(lora, branch) == "target_modules only (selectors ignored)"
    assert _mlx_targets(lora) == "language x attention+mlp"


def test_pre_pr_omitted_selectors_would_have_been_rejected_on_a_vlm():
    """What the flip fixes. Before it, the request model defaulted all three language-side
    selectors False, so an API caller that omitted them reached get_peft_regex with nothing
    selected and got "No layers to finetune" only after the weights were resident."""
    pre_pr = {
        "training_type": "LoRA/QLoRA",
        "finetune_vision_layers": False,
        "finetune_language_layers": False,
        "finetune_attention_modules": False,
        "finetune_mlp_modules": False,
    }

    assert _cuda_guard_fires(pre_pr, "vlm") is True
    assert _cuda_guard_fires(pre_pr, "audio_vlm") is True
    assert _cuda_guard_fires(pre_pr, "text") is False


def test_every_selector_combination_is_covered_by_a_named_case():
    covered = {
        tuple(
            case.get(field, None)
            for field in (
                "finetune_vision_layers",
                "finetune_language_layers",
                "finetune_attention_modules",
                "finetune_mlp_modules",
            )
        )
        for name, case in SELECTOR_CASES.items()
        if name != "omitted"
    }
    all_combinations = set(itertools.product((False, True), repeat = 4))

    assert covered <= all_combinations


@pytest.mark.parametrize("flags", list(itertools.product((False, True), repeat = 4)))
def test_cuda_guard_matches_get_peft_regex_for_the_whole_product(flags):
    """Exhaustive 2^4. get_peft_regex raises unless a layer family AND a module type is on
    (unsloth_zoo/peft_utils.py: "No layers to finetune" / "No modules to finetune"), and the
    guard must fire on exactly that set, never wider."""
    vision, language, attention, mlp = flags
    config = {
        "training_type": "LoRA/QLoRA",
        "finetune_vision_layers": vision,
        "finetune_language_layers": language,
        "finetune_attention_modules": attention,
        "finetune_mlp_modules": mlp,
    }
    get_peft_regex_would_raise = not (vision or language) or not (attention or mlp)

    assert _cuda_guard_fires(config, "vlm") is get_peft_regex_would_raise


# --- defaults for a config that never went through the request model ---


def test_selector_defaults_match_the_cuda_consumer():
    """A config assembled outside the request model (an old job record, the CLI adapter)
    can omit the keys entirely. 4d reads all four with config.get(..., True), so a guard that
    read finetune_vision_layers as False would reject a vision-only run that trains fine."""
    assert _finetune_selectors({}) == (True, True, True, True)


def test_vision_only_run_with_missing_keys_is_not_rejected():
    config = {
        "training_type": "LoRA/QLoRA",
        "finetune_language_layers": False,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": False,
    }

    _check_finetune_targets_after_detect(_Trainer("vlm"), config)


# --- call sites, so deleting the wiring fails a test ---


def _fake_trainer_with_detect(branch: str):
    trainer = _Trainer(branch)
    trainer.pre_detect_calls = []

    def pre_detect_and_load_tokenizer(**kwargs):
        trainer.pre_detect_calls.append(kwargs)

    trainer.pre_detect_and_load_tokenizer = pre_detect_and_load_tokenizer
    return trainer


def test_pre_detect_training_model_runs_the_guard():
    trainer = _fake_trainer_with_detect("vlm")
    config = {
        "training_type": "LoRA/QLoRA",
        "max_seq_length": 2048,
        **SELECTOR_CASES["all_false"],
    }

    with pytest.raises(ValueError, match = "Nothing to train"):
        _pre_detect_training_model(trainer, config, "model", None, "model", False)

    # Detection still ran first: the guard needs the branch it settles.
    assert len(trainer.pre_detect_calls) == 1


def test_pre_detect_training_model_leaves_a_valid_run_alone():
    trainer = _fake_trainer_with_detect("vlm")
    config = {"training_type": "LoRA/QLoRA", "max_seq_length": 2048}

    _pre_detect_training_model(trainer, config, "model", None, "model", False)

    assert len(trainer.pre_detect_calls) == 1


def test_mlx_worker_calls_the_guard_in_its_lora_branch():
    """_run_mlx_training imports mlx, so it cannot be invoked off Apple Silicon. Pin the call
    site structurally instead: inside `if use_lora:` and above the from_pretrained below it."""
    source = textwrap.dedent(inspect.getsource(_run_mlx_training))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_check_mlx_finetune_targets"
    ]

    assert len(calls) == 1

    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "use_lora"
        and any(call in ast.walk(node) for call in calls)
    ]

    assert guarded, "_check_mlx_finetune_targets must sit under `if use_lora:`"

    from_pretrained_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
    ]

    assert from_pretrained_lines
    assert calls[0].lineno < min(from_pretrained_lines)


# --- the guard must never be stricter than the code it guards ---


def test_all_linear_vlm_run_with_the_selectors_off_is_not_rejected():
    """prepare_model_for_training collapses ["all-linear"] to the bare keyword, and
    get_peft_model forces all five selectors True for it, so this trains every linear layer.
    Rejecting it would break a working request, and a resumed run can carry it: target_modules
    is one of the resume structure fields restored from the stored config."""
    for branch in _CUDA_BRANCHES_READING_SELECTORS:
        config = _request_config("LoRA/QLoRA", branch, SELECTOR_CASES["all_false"], ["all-linear"])

        _check_finetune_targets_after_detect(_Trainer(branch), config)


def test_all_linear_as_a_bare_string_is_recognised_too():
    config = {
        "training_type": "LoRA/QLoRA",
        "target_modules": "all-linear",
        **SELECTOR_CASES["all_false"],
    }

    _check_finetune_targets_after_detect(_Trainer("vlm"), config)


def test_all_linear_alongside_other_leaves_is_not_the_keyword():
    """The caller strips "all-linear" out of a longer list and keeps the rest, so the
    selectors do apply and an empty selection is still nothing to train."""
    config = _request_config(
        "LoRA/QLoRA", "vlm", SELECTOR_CASES["all_false"], ["all-linear", "lm_head"]
    )

    assert _requests_all_linear(config) is False
    with pytest.raises(ValueError, match = "Nothing to train"):
        _check_finetune_targets_after_detect(_Trainer("vlm"), config)


@pytest.mark.parametrize("target_modules", [["lm_head"], ["embed_tokens"], ["lm_head", "Wqkv"]])
def test_mlx_keeps_a_target_the_loader_trains_whatever_the_flags_say(target_modules):
    """embed_tokens and lm_head go down get_peft_model's CPT path, applied without consulting
    the layer families. Something trains, so the preflight must not refuse these however the
    four selectors are set."""
    config = _request_config("LoRA/QLoRA", "text", SELECTOR_CASES["all_false"], target_modules)

    _check_mlx_finetune_targets(config)


@pytest.mark.parametrize("target_modules", [["Wqkv"], ["c_fc"], ["all-linear"]])
def test_mlx_refuses_an_all_false_request_whose_targets_need_a_layer_family(target_modules):
    """Surviving the module-type filter is not the same as training.

    These names are not attention or MLP leaves, so get_peft_model's filter keeps them -- but
    the text branch then gates the LoRA application on finetune_language_layers, and with all
    four selectors off the worker's back-fill (which only fires when a module type is on)
    never turns it back on. The run applies no adapters at all: a warning, and a model with
    no trainable parameters. A VLM raises, but only after the weights are loaded."""
    config = _request_config("LoRA/QLoRA", "text", SELECTOR_CASES["all_false"], target_modules)

    with pytest.raises(ValueError, match = "Nothing to train"):
        _check_mlx_finetune_targets(config)


@pytest.mark.parametrize("target_modules", [["Wqkv"], ["c_fc"], ["all-linear"]])
def test_mlx_keeps_those_same_targets_once_a_layer_family_is_on(target_modules):
    """The refusal above is about the layer families, not the names. With
    finetune_language_layers on, the filter keeps these and they train, so refusing here
    would turn a working run away."""
    selectors = {**SELECTOR_CASES["all_false"], "finetune_language_layers": True}
    config = _request_config("LoRA/QLoRA", "text", selectors, target_modules)

    _check_mlx_finetune_targets(config)


def test_mlx_still_rejects_an_empty_module_selection_on_the_defaults():
    config = _request_config("LoRA/QLoRA", "text", SELECTOR_CASES["all_false"], None)

    with pytest.raises(ValueError, match = "Nothing to train"):
        _check_mlx_finetune_targets(config)


def test_the_error_names_every_field_the_caller_has_to_set():
    """An all-false request has to be actionable: the message must name the four request
    fields, not the internal flag names, so an API caller can fix it without reading source."""
    config = _request_config("LoRA/QLoRA", "vlm", SELECTOR_CASES["all_false"], None)

    with pytest.raises(ValueError) as excinfo:
        _check_finetune_targets_after_detect(_Trainer("vlm"), config)

    message = str(excinfo.value)
    for field in TrainingStartRequest.model_fields:
        if field.startswith("finetune_"):
            assert field in message


def test_mlx_reads_the_vision_selector_with_the_mlx_default_not_the_cuda_one():
    """A config that never carried the selectors at all must not be waved through.

    `_finetune_selectors` answers an omitted key with the CUDA consumer's default, and for
    vision that is True. The MLX call site defaults it False and forces it False for a text
    model, so taking True from a missing key would let every config written before these
    fields existed past the guard with nothing to train.
    """
    config = {
        "training_type": "LoRA/QLoRA",
        "target_modules": ["Wqkv"],
        "finetune_language_layers": False,
        "finetune_attention_modules": False,
        "finetune_mlp_modules": False,
        # finetune_vision_layers deliberately absent.
    }
    assert _finetune_selectors(config)[0] is True, "the helper still reports the CUDA default"

    with pytest.raises(ValueError, match = "Nothing to train"):
        _check_mlx_finetune_targets(config)


def test_the_preflight_still_lets_a_vision_only_selection_through():
    """It runs before detection, so it cannot tell a VLM from a text model. A VLM whose
    vision tower is the only thing selected does train, and refusing it here would turn a
    working run away; `_check_mlx_effective_targets` settles it once is_vlm is known."""
    config = {
        "training_type": "LoRA/QLoRA",
        "target_modules": ["Wqkv"],
        "finetune_vision_layers": True,
        "finetune_language_layers": False,
        "finetune_attention_modules": False,
        "finetune_mlp_modules": False,
    }

    _check_mlx_finetune_targets(config)


def test_the_effective_check_refuses_a_text_run_whose_only_selection_was_vision():
    """The call site has already forced vision False for a text model and run the language
    back-fill, so both layer families are off and get_peft_model would apply no adapter at
    all -- a warning, and a model with no trainable parameters."""
    config = {"training_type": "LoRA/QLoRA", "target_modules": ["Wqkv"]}

    with pytest.raises(ValueError, match = "Nothing to train"):
        _check_mlx_effective_targets(config, finetune_language = False, finetune_vision = False)


@pytest.mark.parametrize("language, vision", [(True, False), (False, True), (True, True)])
def test_the_effective_check_passes_whenever_a_layer_family_survives(language, vision):
    config = {"training_type": "LoRA/QLoRA", "target_modules": ["Wqkv"]}

    _check_mlx_effective_targets(config, finetune_language = language, finetune_vision = vision)


@pytest.mark.parametrize(
    "target_modules", [["lm_head"], ["embed_tokens"], ["all-linear", "lm_head"]]
)
def test_the_effective_check_still_spares_a_cpt_target(target_modules):
    """embed_tokens and lm_head train on the CPT path with both layer families off."""
    config = {"training_type": "LoRA/QLoRA", "target_modules": target_modules}

    _check_mlx_effective_targets(config, finetune_language = False, finetune_vision = False)


def test_the_effective_check_runs_after_the_back_fill_at_the_mlx_call_site():
    """Structural. Asked before the back-fill it would refuse runs that go on to train, and
    asked before `finetune_vision` is narrowed by is_vlm it is just the preflight again."""
    import inspect

    from core.training import worker

    source = inspect.getsource(worker)
    call = source.index("_check_mlx_effective_targets(\n            config,")
    backfill = source.index("            finetune_language = True")
    forced = source.index('config.get("finetune_vision_layers", False) if is_vlm else False')
    assert (
        forced < backfill < call
    ), "the effective check must follow both the is_vlm narrowing and the back-fill"
    assert call < source.index("FastMLXModel.get_peft_model("), "and precede the loader"
