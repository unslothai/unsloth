# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Memory tab's Gradient Checkpointing choice must reach the loader and SFTConfig."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import ast  # noqa: E402
import importlib  # noqa: E402
import inspect  # noqa: E402
import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402


_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - stub unusable imports
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

from core.training import trainer as tmod  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


class _Stop(Exception):
    """Unwinds ``_train_worker`` once the SFT config has been captured."""


class _FakeModel:
    config = types.SimpleNamespace()

    def parameters(self):
        return iter(())

    def modules(self):
        return iter(())

    def train(self):
        return self


@pytest.fixture
def run_sft_training(monkeypatch):
    """Drive the real SFT branches of ``_train_worker`` on CPU and report what they built."""
    seen = {"config_args": None, "for_training": []}

    class _FakeVisionModel:
        @staticmethod
        def for_training(model, use_gradient_checkpointing = True):
            seen["for_training"].append(use_gradient_checkpointing)
            return model

        @staticmethod
        def get_peft_model(model, **kwargs):
            return model

    def _capture_sft_config(**config_args):
        seen["config_args"] = config_args
        raise _Stop

    fake_unsloth = types.ModuleType("unsloth")
    fake_unsloth.__spec__ = None
    fake_unsloth.FastModel = _FakeVisionModel
    fake_unsloth.FastVisionModel = _FakeVisionModel
    fake_unsloth.FastLanguageModel = _FakeVisionModel
    fake_unsloth_trainer = types.ModuleType("unsloth.trainer")
    fake_unsloth_trainer.__spec__ = None
    fake_unsloth_trainer.UnslothVisionDataCollator = MagicMock()
    fake_unsloth.trainer = fake_unsloth_trainer

    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    monkeypatch.setitem(sys.modules, "unsloth.trainer", fake_unsloth_trainer)
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
    monkeypatch.setattr(tmod, "FastVisionModel", _FakeVisionModel)
    monkeypatch.setattr(tmod, "FastLanguageModel", _FakeVisionModel)
    monkeypatch.setattr(tmod, "SFTConfig", _capture_sft_config)
    monkeypatch.setattr(tmod, "is_bfloat16_supported", lambda: False)

    def _run(
        gradient_checkpointing,
        *,
        is_audio_vlm = False,
        use_lora = True,
        text = False,
    ):
        t = tmod.UnslothTrainer()
        t.model = _FakeModel()
        t.tokenizer = types.SimpleNamespace()
        t.model_name = "unsloth/Qwen2-VL-7B-Instruct"
        t.is_vlm = not (is_audio_vlm or text)
        t.is_audio_vlm = is_audio_vlm
        # Stands in for the skipped load_model(), which the worker gives the same mode.
        t._use_gradient_checkpointing = tmod.normalize_gradient_checkpointing(
            gradient_checkpointing
        )
        monkeypatch.setattr(
            t,
            "_configure_online_tokenization",
            lambda **kwargs: types.SimpleNamespace(enabled = False),
            raising = True,
        )

        assert t.prepare_model_for_training(
            use_lora = use_lora,
            use_gradient_checkpointing = gradient_checkpointing,
        )
        dataset = [{"text": "hi"}] if text else [{"messages": []}]
        t._train_worker(
            {"dataset": dataset, "final_format": "text" if text else "vision"},
            max_steps = 1,
        )

        assert seen["config_args"] is not None, "the SFT config was never built"
        return seen

    return _run


def test_none_reaches_both_the_sft_config_and_for_training(run_sft_training):
    seen = run_sft_training("none")

    assert (
        seen["config_args"]["gradient_checkpointing"] is False
    ), "the vision branch re-enabled gradient checkpointing after the user turned it off"
    assert (
        "gradient_checkpointing_kwargs" not in seen["config_args"]
    ), "checkpointing kwargs were sent for a run that does not checkpoint"
    assert seen["for_training"] == [False], (
        "for_training re-flagged every layer, leaving them without a "
        "_gradient_checkpointing_func"
    )


@pytest.mark.parametrize("use_lora", [True, False])
def test_audio_vlm_honours_none_as_well(run_sft_training, use_lora):
    seen = run_sft_training("none", is_audio_vlm = True, use_lora = use_lora)

    assert seen["config_args"]["gradient_checkpointing"] is False
    assert "gradient_checkpointing_kwargs" not in seen["config_args"]
    # SFTConfig never turns checkpointing off, so the flags load_model set must be reapplied.
    assert seen["for_training"] == [False], (
        "the audio VLM branch left the module gradient_checkpointing flags alone, "
        "so the run checkpoints despite the Memory tab saying None"
    )


@pytest.mark.parametrize(
    ("choice", "mode"), [("unsloth", "unsloth"), ("true", True), ("", "unsloth")]
)
@pytest.mark.parametrize("is_audio_vlm", [False, True])
def test_the_default_path_still_checkpoints(run_sft_training, choice, mode, is_audio_vlm):
    seen = run_sft_training(choice, is_audio_vlm = is_audio_vlm)

    assert seen["config_args"]["gradient_checkpointing"] is True
    assert seen["config_args"]["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert seen["for_training"] == [mode]


def test_full_finetuning_honours_none_too(run_sft_training):
    seen = run_sft_training("none", use_lora = False)

    assert seen["config_args"]["gradient_checkpointing"] is False
    assert "gradient_checkpointing_kwargs" not in seen["config_args"]
    assert seen["for_training"] == [False]


@pytest.mark.parametrize(("choice", "enabled"), [("none", False), ("unsloth", True)])
@pytest.mark.parametrize("use_lora", [True, False])
def test_a_text_run_carries_the_choice_into_sft_config(run_sft_training, choice, enabled, use_lora):
    """TRL defaults checkpointing on, and train() re-enables it from SFTConfig."""
    seen = run_sft_training(choice, text = True, use_lora = use_lora)

    assert seen["config_args"]["gradient_checkpointing"] is enabled


@pytest.fixture
def load_model_run(monkeypatch):
    """Drive the real ``load_model`` against a stubbed loader and report what it passed."""
    seen = {"loads": [], "for_training": [], "stamp": None}

    def _record_for_training(use_gradient_checkpointing = True):
        seen["for_training"].append(use_gradient_checkpointing)

    class _FakeLoader:
        fail_once_with = None

        @classmethod
        def from_pretrained(cls, **kwargs):
            seen["loads"].append(kwargs)
            if cls.fail_once_with is not None:
                error, cls.fail_once_with = cls.fail_once_with, None
                raise error
            model = _FakeModel()
            # As post_patch_model does; real text LoRA loads record it in get_peft_model instead.
            seen["stamp"] = kwargs.get("use_gradient_checkpointing")
            model._unsloth_gradient_checkpointing = seen["stamp"]
            model.for_training = _record_for_training
            return model, types.SimpleNamespace(image_processor = object())

        @staticmethod
        def get_peft_model(
            model,
            use_gradient_checkpointing = "unsloth",
            **kwargs,
        ):
            # Adding adapters re-records the mode.
            seen["stamp"] = use_gradient_checkpointing
            model._unsloth_gradient_checkpointing = use_gradient_checkpointing
            return model

    fake_unsloth = types.ModuleType("unsloth")
    fake_unsloth.__spec__ = None
    fake_unsloth.FastModel = _FakeLoader
    fake_unsloth.FastVisionModel = _FakeLoader
    fake_unsloth.FastLanguageModel = _FakeLoader

    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
    monkeypatch.setattr(tmod, "FastVisionModel", _FakeLoader)
    monkeypatch.setattr(tmod, "FastLanguageModel", _FakeLoader)
    monkeypatch.setattr(tmod, "clear_gpu_cache", lambda *a, **k: None)
    monkeypatch.setattr(tmod, "get_device_map", lambda *a, **k: "cuda:0")
    monkeypatch.setattr(tmod, "get_visible_gpu_count", lambda *a, **k: 1)
    monkeypatch.setattr(tmod, "raise_if_offloaded", lambda *a, **k: None)
    monkeypatch.setattr(tmod, "restore_hf_cache_repo_identity", lambda *a, **k: None)
    monkeypatch.setattr(tmod, "is_bfloat16_supported", lambda: False)

    def _run(
        gradient_checkpointing,
        *,
        full_finetuning = False,
        kind = "vision",
        fail_once = None,
    ):
        audio_type = "audio_vlm" if kind == "audio_vlm" else None
        monkeypatch.setattr(tmod, "detect_audio_type_checked", lambda *a, **k: (audio_type, True))
        monkeypatch.setattr(tmod, "is_vision_model", lambda *a, **k: kind == "vision")
        _FakeLoader.fail_once_with = fail_once

        t = tmod.UnslothTrainer()
        monkeypatch.setattr(t, "_cleanup_audio_artifacts", lambda: None, raising = True)
        assert t.load_model(
            model_name = "unsloth/Qwen2-VL-7B-Instruct",
            max_seq_length = 2048,
            is_dataset_image = kind == "vision",
            is_dataset_audio = kind == "audio_vlm",
            local_files_only = True,
            full_finetuning = full_finetuning,
            use_gradient_checkpointing = gradient_checkpointing,
        )
        return t, seen

    return _run


@pytest.mark.parametrize(
    ("choice", "mode"), [("none", False), ("true", True), ("unsloth", "unsloth")]
)
@pytest.mark.parametrize("kind", ["vision", "audio_vlm", "text"])
@pytest.mark.parametrize("full_finetuning", [True, False])
def test_the_loader_is_told_the_choice(load_model_run, kind, full_finetuning, choice, mode):
    """A full finetune never revisits the loaded mode, which must be True, False or "unsloth"."""
    _, seen = load_model_run(choice, full_finetuning = full_finetuning, kind = kind)

    assert seen["loads"][-1]["use_gradient_checkpointing"] is mode, (
        "the model was not loaded with the chosen mode, so the trainer restores another one "
        "at train() and the Memory tab is ignored"
    )
    if full_finetuning:
        assert seen["for_training"] == [mode]


def test_the_retry_after_a_source_code_failure_keeps_the_choice(load_model_run):
    """The retry reloads the model, so it re-records the mode and must not lose it."""
    _, seen = load_model_run(
        "none",
        full_finetuning = True,
        fail_once = OSError("could not get source code"),
    )

    assert len(seen["loads"]) == 2, "the retry never ran, so this asserts nothing"
    assert seen["loads"][-1]["use_gradient_checkpointing"] is False


@pytest.mark.parametrize("kind", ["vision", "audio_vlm", "text"])
@pytest.mark.parametrize("use_lora", [True, False])
def test_a_later_mode_applies_only_where_adapters_are_added(
    load_model_run, monkeypatch, use_lora, kind
):
    """LoRA re-applies a prepare-time mode; a full finetune keeps the loaded one."""
    t, seen = load_model_run("unsloth", full_finetuning = not use_lora, kind = kind)
    warnings = []
    monkeypatch.setattr(tmod.logger, "warning", lambda message, *a, **k: warnings.append(message))

    assert t.prepare_model_for_training(use_lora = use_lora, use_gradient_checkpointing = "none")

    expected = False if use_lora else "unsloth"
    assert t._use_gradient_checkpointing == expected
    assert seen["stamp"] == expected
    assert any("load_model" in w for w in warnings) is not use_lora


def _cli_load_model_calls():
    cli = _BACKEND.parent.parent / "unsloth_cli" / "commands" / "train.py"
    tree = ast.parse(cli.read_text(encoding = "utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_model"
        and getattr(node.func.value, "id", None) == "trainer"
    ]
    assert calls, "the CLI no longer loads a model here"
    return calls


def test_the_cli_chooses_the_mode_when_it_loads():
    """The CLI must load with the configured mode, and load a full finetune as one."""
    calls = _cli_load_model_calls()

    for node in calls:
        forwarded = [k for k in node.keywords if k.arg == "use_gradient_checkpointing"]
        assert forwarded, (
            f"train.py line {node.lineno} loads without the configured gradient checkpointing, "
            "so the run is built with a different mode"
        )
        value = forwarded[0].value
        assert (
            ast.unparse(value) == "cfg.training.gradient_checkpointing"
        ), f"train.py line {node.lineno} does not pass the configured mode through unchanged"
        full = [k.value for k in node.keywords if k.arg == "full_finetuning"]
        assert (
            full and ast.unparse(full[0]) == "not use_lora"
        ), f"train.py line {node.lineno} loads a full finetune as if adapters will follow"


def test_both_trainers_accept_what_the_cli_loads_with():
    """The CLI builds either trainer from one call site, so its keywords must bind to both."""
    from core.training.training import _MLXTrainerAdapter

    keyword_sets = [{k.arg for k in node.keywords if k.arg} for node in _cli_load_model_calls()]

    for trainer_class in (tmod.UnslothTrainer, _MLXTrainerAdapter):
        signature = inspect.signature(trainer_class.load_model)
        for keywords in keyword_sets:
            signature.bind(object(), **{name: None for name in keywords})


def test_the_mlx_adapter_keeps_the_mode_it_was_loaded_with():
    """The PEFT entry wins in the worker config, so an omitted mode keeps the loaded one."""
    from core.training.training import _MLXTrainerAdapter

    adapter = _MLXTrainerAdapter()
    assert adapter.load_model(model_name = "unsloth/Qwen3-4B", use_gradient_checkpointing = False)
    assert adapter.prepare_model_for_training(use_lora = True)

    assert adapter._peft_config["gradient_checkpointing"] is False
    assert adapter._model_config["gradient_checkpointing"] is False


def test_the_mlx_adapter_rejects_load_arguments_it_would_drop():
    """An unknown load keyword must raise rather than silently never reach the worker."""
    from core.training.training import _MLXTrainerAdapter

    signature = inspect.signature(_MLXTrainerAdapter.load_model)

    with pytest.raises(TypeError):
        signature.bind(object(), model_name = "unsloth/Qwen3-4B", model_revision = "abc123")


@pytest.mark.parametrize("full_finetuning", [True, False])
def test_omitting_the_argument_keeps_the_choice(load_model_run, full_finetuning):
    t, _ = load_model_run("none", full_finetuning = full_finetuning)

    assert t.prepare_model_for_training(use_lora = not full_finetuning)

    assert t.model._unsloth_gradient_checkpointing is False
    assert t._use_gradient_checkpointing is False


def _trainer_calls_to(method):
    worker = _BACKEND / "core" / "training" / "worker.py"
    tree = ast.parse(worker.read_text(encoding = "utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and getattr(node.func.value, "id", None) == "trainer"
    ]


def _reads_the_memory_tab(keyword):
    """Old configs lack the key, so the ``"unsloth"`` fallback is part of the contract."""
    node = keyword.value
    if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "get"):
        return False
    if getattr(node.func.value, "id", None) != "config":
        return False
    return [getattr(a, "value", None) for a in node.args] == [
        "gradient_checkpointing",
        "unsloth",
    ]


@pytest.mark.parametrize(
    ("method", "expected"),
    [("prepare_model_for_training", 3), ("load_model", 2)],
)
def test_the_worker_forwards_the_setting_on_every_call(method, expected):
    """An omitted or literal argument ignores the Memory tab."""
    calls = _trainer_calls_to(method)

    assert (
        len(calls) >= expected
    ), f"expected at least {expected} trainer.{method} calls, found {len(calls)}"
    unforwarded = [
        node.lineno
        for node in calls
        if not any(
            k.arg == "use_gradient_checkpointing" and _reads_the_memory_tab(k)
            for k in node.keywords
        )
    ]
    assert not unforwarded, (
        f"worker.py lines {unforwarded} call {method} without forwarding the config's "
        "gradient_checkpointing; that run ignores the Memory tab and checkpoints anyway. "
        "Rewriting that expression means teaching this check the new shape, not dropping it."
    )
