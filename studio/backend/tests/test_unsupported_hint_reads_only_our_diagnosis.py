# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The "not supported yet" rewrite must not read the child's own output.

A llama-server failure now carries a tail of the child's stdout. The route
scans the whole error string for phrases like "does not support", so once that
tail was attached, any llama.cpp line containing one of them rewrote the
diagnosis to "This model is not supported yet. Try a different model." -- for a
failure that had nothing to do with the checkpoint. llama.cpp really does print
such lines, for instance when a Vulkan device lacks 16-bit storage, so this was
reachable on Linux and Windows as much as on macOS.

The evidence still reaches the user; it just no longer votes on the diagnosis.
"""

from __future__ import annotations

import pytest

from core.inference.llama_cpp import LlamaCppBackend
from routes.inference import (
    _diagnosis_text,
    _is_unsupported_nvfp4_inference_error,
    _maybe_unsupported_message,
)

REWRITE = "This model is not supported yet"


def _failure_carrying(child_output: str) -> str:
    """The API error for a llama-server death whose output is ``child_output``."""
    return LlamaCppBackend._classify_llama_start_failure(
        child_output,
        "/m.gguf",
        "u/x",
        1,
        None,
        "/tmp/llama-1-port-8080.log",
        (),
    )


@pytest.mark.parametrize(
    "child_line",
    [
        "ggml_vulkan: device Intel(R) UHD does not support 16-bit storage",
        "vulkan: this device is not supported by the build",
        "loader: No config file found in the model directory",
        "warning: flash attention is not yet supported on this backend",
    ],
)
def test_a_phrase_in_the_childs_output_does_not_rewrite_the_diagnosis(child_line):
    msg = _failure_carrying(child_line + "\nabort\n")
    assert child_line in msg, "the tail should still be shown to the user"
    assert not _maybe_unsupported_message(msg).startswith(REWRITE)


def test_the_same_phrase_in_our_own_text_still_rewrites():
    ours = "Model architecture 'foo' is not supported by this llama.cpp build."
    assert _maybe_unsupported_message(ours).startswith(REWRITE)


def test_our_text_still_rewrites_even_when_a_tail_is_attached():
    ours = "Model architecture 'foo' is not supported by this llama.cpp build."
    withtail = ours + "\n\nllama-server output:\nnoise\n\nFull log: /tmp/l.log"
    out = _maybe_unsupported_message(withtail)
    assert out.startswith(REWRITE)
    # The rewrite quotes the whole original, evidence included.
    assert "llama-server output:" in out and "Full log:" in out


def test_the_nvfp4_matcher_also_ignores_the_tail():
    ours = "NVFP4 checkpoint with per-module MLX quantization metadata"
    assert _is_unsupported_nvfp4_inference_error(ours)
    tail = (
        "llama-server failed to start.\n\nllama-server output:\n"
        "nvfp4 per-module mlx quantization metadata\n"
    )
    assert not _is_unsupported_nvfp4_inference_error(tail)


@pytest.mark.parametrize(
    "msg",
    [
        "",
        "some unrelated backend error",
        "a message mentioning llama-server output: inline, not as a block",
        "trailing marker only\n\nFull log: /tmp/l.log",
    ],
)
def test_messages_without_a_diagnostics_block_are_untouched(msg):
    """Every other error source must behave exactly as it did before."""
    expected = msg.split("\n\nFull log: ")[0]
    assert _diagnosis_text(msg) == expected
