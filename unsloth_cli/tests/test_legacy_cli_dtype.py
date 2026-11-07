# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth-cli.py documents `--dtype None` for auto detection, but the loader wants a real None."""

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
        "unsloth_legacy_cli_dtype", _REPO_ROOT / "unsloth-cli.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module(name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


@pytest.mark.parametrize(
    "spelling, expected",
    [("None", None), ("none", None), ("auto", None), ("", None), ("bfloat16", "bfloat16")],
)
def test_dtype_reaches_the_loader_as_none_or_unchanged(monkeypatch, spelling, expected):
    seen = {}

    class FastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
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

    legacy_cli = _load_legacy_cli()
    args = types.SimpleNamespace(
        model_name = "unsloth/llama-3-8b",
        max_seq_length = 2048,
        dtype = spelling,
        load_in_4bit = True,
        use_dora = False,
    )
    with pytest.raises(_Stop):
        legacy_cli.run(args)
    assert seen["dtype"] == expected
