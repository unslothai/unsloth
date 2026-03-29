# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from core.training.sft_meta_guard import (
    inspect_meta_backed_embeddings,
    maybe_enable_sft_meta_guard,
)


class _FakeEmbeddingModule:
    def __init__(self, is_meta: bool):
        self.weight = SimpleNamespace(is_meta = is_meta)


class _FakeModel:
    def __init__(self, *, input_is_meta: bool, output_is_meta: bool):
        self.config = SimpleNamespace(_name_or_path = "unsloth/Qwen3.5-27B")
        self.hf_device_map = {"model.embed_tokens": 0}
        self._input_module = _FakeEmbeddingModule(input_is_meta)
        self._output_module = _FakeEmbeddingModule(output_is_meta)

    def get_input_embeddings(self):
        return self._input_module

    def get_output_embeddings(self):
        return self._output_module


def test_inspect_meta_backed_embeddings_detects_meta_weights():
    model = _FakeModel(input_is_meta = True, output_is_meta = False)
    assert inspect_meta_backed_embeddings(model) == ["input_embeddings=meta"]


def test_maybe_enable_sft_meta_guard_extends_ignored_names():
    model = _FakeModel(input_is_meta = False, output_is_meta = True)
    logger = MagicMock()
    old_value = os.environ.get("UNSLOTH_IGNORED_TOKENIZER_NAMES")
    os.environ["UNSLOTH_IGNORED_TOKENIZER_NAMES"] = "existing/model"
    try:
        assert maybe_enable_sft_meta_guard(model, "unsloth/Qwen3.5-27B", logger) is True
        ignored = os.environ["UNSLOTH_IGNORED_TOKENIZER_NAMES"].split("\n")
        assert "existing/model" in ignored
        assert "unsloth/Qwen3.5-27B" in ignored
        assert "unsloth/qwen3.5-27b" in ignored
        logger.warning.assert_called_once()
    finally:
        if old_value is None:
            os.environ.pop("UNSLOTH_IGNORED_TOKENIZER_NAMES", None)
        else:
            os.environ["UNSLOTH_IGNORED_TOKENIZER_NAMES"] = old_value


def test_maybe_enable_sft_meta_guard_noops_for_materialized_model():
    model = _FakeModel(input_is_meta = False, output_is_meta = False)
    logger = MagicMock()
    old_value = os.environ.get("UNSLOTH_IGNORED_TOKENIZER_NAMES")
    os.environ["UNSLOTH_IGNORED_TOKENIZER_NAMES"] = "keep/me"
    try:
        assert (
            maybe_enable_sft_meta_guard(model, "unsloth/Qwen3.5-27B", logger) is False
        )
        assert os.environ["UNSLOTH_IGNORED_TOKENIZER_NAMES"] == "keep/me"
        logger.warning.assert_not_called()
    finally:
        if old_value is None:
            os.environ.pop("UNSLOTH_IGNORED_TOKENIZER_NAMES", None)
        else:
            os.environ["UNSLOTH_IGNORED_TOKENIZER_NAMES"] = old_value
