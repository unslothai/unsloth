# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ``unsloth_<model>_<timestamp>`` directory-name fallback for base model detection.

It is the last resort in ``get_base_model_from_checkpoint`` / ``get_base_model_from_lora``,
reached only when no config names a base model. It used to accept a two-segment
``unsloth_<model>`` name, slice the model part away to nothing, and return the bogus repo id
``unsloth/`` -- which the export path and ``/models/lora/base`` then handed to the Hub.
"""

import json
import sys
import types

import pytest

# Keep this test runnable where optional logging deps are not installed.
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

from utils.models.model_config import (  # noqa: E402
    _base_model_from_dir_name,
    get_base_model_from_checkpoint,
    get_base_model_from_lora,
)


@pytest.mark.parametrize(
    "dir_name,expected",
    [
        # The shape the heuristic is written for.
        ("unsloth_Qwen3-8B_20260101-120000", "unsloth/Qwen3-8B"),
        # Underscores inside the model name survive the round trip.
        ("unsloth_llama_3_8b_20260101", "unsloth/llama_3_8b"),
        # No model segment between the prefix and the timestamp: the reported bug.
        ("unsloth_Qwen3-8B", None),
        ("unsloth_", None),
        ("unsloth__20260101", None),
        # A doubled separator must not leak a leading underscore into the repo id.
        ("unsloth__Qwen3-8B_20260101", "unsloth/Qwen3-8B"),
        # Not ours: leave it to the caller's "could not detect" path.
        ("my-finetune_20260101", None),
        ("", None),
    ],
)
def test_base_model_from_dir_name(dir_name, expected):
    assert _base_model_from_dir_name(dir_name) == expected


def _write_adapter(directory):
    directory.mkdir(parents = True)
    # No base_model_name_or_path, so detection has to fall through to the directory name.
    (directory / "adapter_config.json").write_text(json.dumps({}), encoding = "utf-8")
    (directory / "adapter_model.safetensors").write_bytes(b"")
    return directory


def test_lora_without_timestamp_does_not_report_the_bare_org(tmp_path):
    adapter = _write_adapter(tmp_path / "unsloth_Qwen3-8B")
    assert get_base_model_from_lora(str(adapter)) is None


def test_lora_with_timestamp_still_resolves(tmp_path):
    adapter = _write_adapter(tmp_path / "unsloth_Qwen3-8B_20260101-120000")
    assert get_base_model_from_lora(str(adapter)) == "unsloth/Qwen3-8B"


def test_checkpoint_without_timestamp_does_not_report_the_bare_org(tmp_path):
    checkpoint = tmp_path / "unsloth_Qwen3-8B"
    checkpoint.mkdir()
    assert get_base_model_from_checkpoint(str(checkpoint)) is None


def test_checkpoint_with_timestamp_still_resolves(tmp_path):
    checkpoint = tmp_path / "unsloth_Qwen3-8B_20260101-120000"
    checkpoint.mkdir()
    assert get_base_model_from_checkpoint(str(checkpoint)) == "unsloth/Qwen3-8B"
