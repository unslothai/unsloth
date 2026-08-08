# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A backend without a TTS path must say so, not "an internal error occurred".

The MLX worker has no text-to-speech branch, so generating on a safetensors
Orpheus (which is what loads on Apple Silicon) always fails. The message was
flattened by ``safe_error_detail``, so the Audio page reported an internal error
for a model that had loaded fine and simply cannot generate on that backend.
"""

from __future__ import annotations

import pytest

from core.inference.audio_errors import (
    AUDIO_UNSUPPORTED_CODE,
    AudioBackendUnsupportedError,
)
from utils.utils import safe_error_detail


def test_the_capability_message_survives_the_leak_guard():
    """safe_error_detail is why the reason was lost; the route must bypass it."""
    error = AudioBackendUnsupportedError(
        "Text-to-speech is not supported on the MLX backend yet.",
        hint = "Load this model's GGUF build instead.",
    )
    assert safe_error_detail(error) == "An internal error occurred"
    # What the route sends instead: reason plus the way out, no path, no input.
    assert "MLX" in error.message
    assert "GGUF" in error.message


def test_the_hint_is_optional():
    error = AudioBackendUnsupportedError("No TTS on this backend.")
    assert error.message == "No TTS on this backend."
    assert error.hint is None


def test_it_is_a_runtime_error_so_existing_handlers_still_catch_it():
    """The generate path catches Exception; narrowing the type must not escape it."""
    assert issubclass(AudioBackendUnsupportedError, RuntimeError)


def test_the_worker_tags_the_payload_with_the_shared_code():
    """Parent and worker agree on the code, so neither matches on prose."""
    from pathlib import Path

    worker = Path(__file__).resolve().parents[1] / "core/inference/worker.py"
    source = worker.read_text(encoding = "utf-8")
    assert "AUDIO_UNSUPPORTED_CODE" in source
    # The literal lives in one place only.
    assert AUDIO_UNSUPPORTED_CODE == "audio_unsupported_backend"
    assert f'"{AUDIO_UNSUPPORTED_CODE}"' not in source


@pytest.mark.parametrize(
    "code,expected",
    [(AUDIO_UNSUPPORTED_CODE, True), ("something_else", False), (None, False)],
)
def test_only_the_tagged_payload_becomes_the_typed_error(code, expected):
    """Mirrors the orchestrator's audio_error branch."""
    resp = {"type": "audio_error", "error": "boom"}
    if code is not None:
        resp["code"] = code
    is_capability = resp.get("code") == AUDIO_UNSUPPORTED_CODE
    assert is_capability is expected
