# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Only an unknown architecture blames the prebuilt; a drafter context that could not be created is memory."""

import pytest

from core.inference.llama_cpp import _spec_start_failure


def test_unknown_architecture_points_at_an_update():
    reason, code = _spec_start_failure(
        "llama_model_load: error loading model: unknown model architecture: 'qwen38mtp'"
    )
    assert code == "binary_outdated"
    assert "unsloth studio update" in reason


@pytest.mark.parametrize(
    "line",
    [
        "common_init_from_params: failed to create mtp context",
        "llama_init_from_model: failed to create llama_context",
        "srv load_model: failed to measure mtp context memory",
        "Failed to measure draft model memory",
    ],
)
def test_a_context_that_could_not_be_created_does_not_send_users_to_update(line):
    reason, code = _spec_start_failure(
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 2048.00 MiB on device 1: cudaMalloc failed: out of memory\n"
        + line
    )
    assert code == "runtime_error"
    assert "update" not in reason
    assert "memory" in reason


def test_any_other_crash_keeps_the_generic_retry():
    reason, code = _spec_start_failure("Segmentation fault")
    assert code == "runtime_error"
    assert reason == "retrying without speculative decoding in case the drafter is the cause"
