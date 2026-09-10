# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the vision-mmproj batch/ubatch default."""

import inspect

import pytest

from studio.backend.core.inference.llama_cpp import (
    LlamaCppBackend,
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

    def test_vision_switched_off_is_unchanged(self):
        # The projector is dropped before launch, so the bigger micro-batch would
        # reserve compute buffer for images the child cannot be sent.
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(
            "mmproj-F16.gguf",
            None,
            None,
            None,
            is_vision = True,
            disable_vision = True,
        )
        assert n_batch is None
        assert n_ubatch is None

    @pytest.mark.parametrize("flag", ["--no-mmproj", "--no-mmproj=true"])
    def test_extra_arg_no_mmproj_is_unchanged(self, flag):
        n_batch, n_ubatch = _batch_ubatch_for_mmproj(
            "mmproj-F16.gguf",
            None,
            None,
            [flag],
            is_vision = True,
        )
        assert n_batch is None
        assert n_ubatch is None


def test_default_is_applied_after_the_companion_download():
    """A Hub load carries no mmproj_path of its own.

    ``_resolve_gguf_load_intent`` leaves ``intent.mmproj_path`` unset for a repo id and
    ``_download_mmproj`` is what assigns it, so deciding at the intent unpack reads None
    on the ordinary loading path and never raises the sizes at all. The fit has to see
    the decision too: it prices the compute buffer off the micro-batch that launches.
    """
    source = inspect.getsource(LlamaCppBackend.load_model)
    download = source.index("self._download_mmproj(")
    decide = source.index("_batch_ubatch_for_mmproj(")
    price = source.index("_ubatch_for_slots(n_parallel)")
    assert download < decide < price
