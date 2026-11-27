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


_TARGET = "/hf/snapshots/abc/Model-UD-Q4_K_XL.gguf"
_DRAFTER = "/hf/snapshots/abc/dspark-Model-Q8_0.gguf"
_TARGET_OOM = (
    "E ggml_backend_cuda_buffer_type_alloc_buffer: allocating 112679.75 MiB on device 0: cudaMalloc failed: out of memory\n"
    "E graph_reserve: failed to allocate compute buffers\n"
    "E llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers\n"
    f"E cmn  common_init_: failed to create context with model '{_TARGET}'\n"
)


def test_a_target_context_that_failed_before_the_drafter_loaded_keeps_the_drafter():
    from core.inference.llama_cpp import _target_failed_before_drafter
    assert _target_failed_before_drafter(_TARGET_OOM, _DRAFTER)


def test_a_failure_after_the_drafter_started_loading_still_drops_it():
    from core.inference.llama_cpp import _target_failed_before_drafter

    out = (
        "I cmn  common_init_: warming up the model with an empty run\n"
        f"I common_speculative_init_result: loading draft model '{_DRAFTER}'\n"
        f"E cmn  common_init_: failed to create context with model '{_DRAFTER}'\n"
    )
    assert not _target_failed_before_drafter(out, _DRAFTER)
    # The drafter's own context failing is the drafter's fault even if the load line scrolled out.
    assert not _target_failed_before_drafter(out.split("\n", 2)[2], _DRAFTER)


@pytest.mark.parametrize("out", ["Segmentation fault", "failed to create mtp context"])
def test_other_failures_and_embedded_heads_keep_the_drafter_retry(out):
    from core.inference.llama_cpp import _target_failed_before_drafter
    assert not _target_failed_before_drafter(out, _DRAFTER)
    assert not _target_failed_before_drafter(_TARGET_OOM, None)
