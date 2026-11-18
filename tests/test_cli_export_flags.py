# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth-cli.py gated every export behind --save_model, so --save_gguf, --push_gguf and
--push_model alone finished training and silently skipped the requested export."""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


class _RecordingModel:
    def __init__(self, calls):
        self._calls = calls

    def save_pretrained_gguf(self, *args, **kwargs):
        self._calls.append(("save_pretrained_gguf", args, kwargs))

    def push_to_hub_gguf(self, *args, **kwargs):
        self._calls.append(("push_to_hub_gguf", args, kwargs))

    def save_pretrained_merged(self, *args, **kwargs):
        self._calls.append(("save_pretrained_merged", args, kwargs))

    def push_to_hub_merged(self, *args, **kwargs):
        self._calls.append(("push_to_hub_merged", args, kwargs))


def _load_legacy_cli():
    spec = importlib.util.spec_from_file_location(
        "unsloth_legacy_cli_export_flags", _REPO_ROOT / "unsloth-cli.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _run_cli(monkeypatch, **overrides):
    calls = []

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            return _RecordingModel(calls), types.SimpleNamespace(eos_token = "</s>")

        @staticmethod
        def get_peft_model(model, **kwargs):
            return model

    class _Dataset:
        def map(
            self,
            func,
            batched = False,
        ):
            return self

    class _Trainer:
        distributed_world_size = 1

        def __init__(self, **kwargs):
            pass

        def train(self):
            calls.append(("train", (), {}))

    for name, module in {
        "unsloth": _module(
            "unsloth",
            FastLanguageModel = FastLanguageModel,
            is_bfloat16_supported = lambda: True,
            RawTextDataLoader = object,
        ),
        "unsloth.models": _module("unsloth.models"),
        "unsloth.models.loader_utils": _module(
            "unsloth.models.loader_utils", prepare_device_map = lambda: (None, False)
        ),
        "datasets": _module("datasets", load_dataset = lambda *a, **k: _Dataset()),
        "transformers": _module("transformers"),
        "transformers.utils": _module(
            "transformers.utils",
            strtobool = lambda v: str(v).strip().lower() in ("1", "true", "yes", "y", "t", "on"),
        ),
        "trl": _module(
            "trl",
            SFTTrainer = _Trainer,
            SFTConfig = lambda **kwargs: types.SimpleNamespace(**kwargs),
        ),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    args = dict(
        model_name = "m",
        max_seq_length = 32,
        dtype = None,
        load_in_4bit = False,
        dataset = "unused",
        raw_text_file = None,
        chunk_size = 32,
        stride = 8,
        r = 8,
        lora_alpha = 8,
        lora_dropout = 0.0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 0,
        use_rslora = False,
        loftq_config = None,
        use_dora = False,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 4,
        gradient_accumulation_steps = 1,
        warmup_steps = 0,
        max_steps = 1,
        learning_rate = 1e-4,
        logging_steps = 1,
        optim = "adamw_torch",
        weight_decay = 0.0,
        lr_scheduler_type = "linear",
        seed = 0,
        output_dir = "unused",
        report_to = "none",
        packing = False,
        context_parallel_size = 1,
        save_model = False,
        save_method = "merged_16bit",
        save_gguf = False,
        save_path = "model",
        quantization = ["q8_0"],
        push_model = False,
        push_gguf = False,
        hub_path = "hf/model",
        hub_token = "tok",
    )
    args.update(overrides)
    _load_legacy_cli().run(argparse.Namespace(**args))
    return [(name, a, k) for name, a, k in calls if name != "train"], any(
        name == "train" for name, _, _ in calls
    )


@pytest.mark.parametrize(
    "overrides, expected",
    [
        (dict(save_gguf = True), ["save_pretrained_gguf"]),
        (dict(push_gguf = True), ["push_to_hub_gguf"]),
        (dict(push_model = True), ["push_to_hub_merged"]),
        (dict(save_model = True), ["save_pretrained_merged"]),
        (
            dict(save_model = True, push_model = True),
            ["save_pretrained_merged", "push_to_hub_merged"],
        ),
        (
            dict(save_gguf = True, push_gguf = True),
            ["save_pretrained_gguf", "push_to_hub_gguf"],
        ),
        (
            dict(save_model = True, save_gguf = True, push_model = True),
            ["save_pretrained_gguf", "push_to_hub_gguf"],
        ),
        (dict(), []),
    ],
)
def test_export_flags_dispatch_after_training(monkeypatch, overrides, expected):
    calls, trained = _run_cli(monkeypatch, **overrides)
    assert trained
    assert [name for name, _, _ in calls] == expected


def test_push_gguf_alone_routes_hub_path_and_token(monkeypatch):
    calls, _ = _run_cli(monkeypatch, push_gguf = True)
    name, args, kwargs = calls[0]
    assert name == "push_to_hub_gguf"
    assert args[0] == "hf/model"
    assert kwargs["token"] == "tok"
    assert kwargs["quantization_method"] == "q8_0"


def test_push_model_alone_routes_hub_path_token_and_save_method(monkeypatch):
    calls, _ = _run_cli(monkeypatch, push_model = True, save_method = "lora")
    name, args, kwargs = calls[0]
    assert name == "push_to_hub_merged"
    assert args[0] == "hf/model"
    assert args[2] == "lora"
    assert kwargs["token"] == "tok"


def test_push_model_alone_does_not_save_merged_locally(monkeypatch):
    calls, _ = _run_cli(monkeypatch, push_model = True)
    assert "save_pretrained_merged" not in [name for name, _, _ in calls]


def test_save_gguf_saves_once_per_quantization_method(monkeypatch):
    calls, _ = _run_cli(monkeypatch, save_gguf = True, quantization = ["f16", "q4_k_m", "q8_0"])
    methods = [kwargs["quantization_method"] for _, _, kwargs in calls]
    assert methods == ["f16", "q4_k_m", "q8_0"]


def test_no_export_flags_warns_and_exports_nothing(monkeypatch, capsys):
    calls, trained = _run_cli(monkeypatch)
    assert trained
    assert calls == []
    assert "not saved" in capsys.readouterr().out
