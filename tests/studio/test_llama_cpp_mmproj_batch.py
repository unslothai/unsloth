# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the vision-mmproj batch/ubatch default."""

import pytest

from studio.backend.core.inference.llama_cpp import (
    _batch_ubatch_for_mmproj,
    _MMPROJ_DEFAULT_N_BATCH_UBATCH,
)


class TestBatchUbatchForMmproj:
    """Tests for _batch_ubatch_for_mmproj."""

    def test_vision_mmproj_without_override_gets_default(self):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(
            "mmproj-F16.gguf",
            None,
            None,
            None,
            is_vision = True,
        )
        assert n_batch == _MMPROJ_DEFAULT_N_BATCH_UBATCH
        assert n_ubatch == _MMPROJ_DEFAULT_N_BATCH_UBATCH

    def test_non_vision_mmproj_is_unchanged(self):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(
            "mmproj-F16.gguf",
            None,
            None,
            None,
            is_vision = False,
        )
        assert n_batch is None
        assert n_ubatch is None

    def test_explicit_n_batch_is_preserved(self):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(
            "mmproj-F16.gguf",
            1024,
            None,
            None,
            is_vision = True,
        )
        assert n_batch == 1024
        assert n_ubatch is None

    def test_explicit_n_ubatch_is_preserved(self):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(
            "mmproj-F16.gguf",
            None,
            1024,
            None,
            is_vision = True,
        )
        assert n_batch is None
        assert n_ubatch == 1024

    def test_extra_arg_batch_override_is_respected(self):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(
            "mmproj-F16.gguf",
            None,
            None,
            ["--batch-size", "1024"],
            is_vision = True,
        )
        assert n_batch is None
        assert n_ubatch is None

    def test_extra_arg_ubatch_override_is_respected(self):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(
            "mmproj-F16.gguf",
            None,
            None,
            ["--ubatch-size", "1024"],
            is_vision = True,
        )
        assert n_batch is None
        assert n_ubatch is None
