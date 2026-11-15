# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth-cli.py reads --use_gradient_checkpointing as a string, but get_peft_model wants True, False or "unsloth"."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _Stop(Exception):
    pass


def _load_legacy_cli():
    spec = importlib.util.spec_from_file_location(
        "unsloth_legacy_cli_gradient_checkpointing", _REPO_ROOT / "unsloth-cli.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _setup(monkeypatch, spelling):
    seen = {}

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            seen["loaded"] = True
            return object(), types.SimpleNamespace(eos_token = "</s>")

        @staticmethod
        def get_peft_model(model, **kwargs):
            seen.update(kwargs)
            raise _Stop

    for name, module in {
        "unsloth": _module(
            "unsloth",
            FastLanguageModel = FastLanguageModel,
            is_bfloat16_supported = lambda: False,
            RawTextDataLoader = object,
        ),
        "unsloth.models": _module("unsloth.models"),
        "unsloth.models.loader_utils": _module(
            "unsloth.models.loader_utils", prepare_device_map = lambda: (None, False)
        ),
        "datasets": _module("datasets", load_dataset = None),
        "transformers": _module("transformers"),
        "transformers.utils": _module("transformers.utils", strtobool = None),
        "trl": _module("trl", SFTTrainer = None, SFTConfig = None),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    args = types.SimpleNamespace(
        model_name = "unsloth/llama-3-8b",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
        use_dora = False,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = spelling,
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    return _load_legacy_cli(), args, seen


@pytest.mark.parametrize(
    "spelling, expected",
    [
        ("unsloth", "unsloth"),
        ("Unsloth", "unsloth"),
        ("", "unsloth"),
        ("mlx", "unsloth"),
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("none", False),
        ("off", False),
    ],
)
def test_gradient_checkpointing_reaches_get_peft_model_as_a_supported_value(
    monkeypatch, spelling, expected
):
    legacy_cli, args, seen = _setup(monkeypatch, spelling)
    with pytest.raises(_Stop):
        legacy_cli.run(args)
    got = seen["use_gradient_checkpointing"]
    assert type(got) is type(expected) and got == expected


def test_unknown_gradient_checkpointing_fails_before_the_model_loads(monkeypatch):
    legacy_cli, args, seen = _setup(monkeypatch, "flase")
    with pytest.raises(ValueError, match = "unsloth, true or false"):
        legacy_cli.run(args)
    assert "loaded" not in seen
