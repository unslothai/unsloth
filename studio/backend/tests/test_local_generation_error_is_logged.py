# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A local generation failure must leave the cause in the server log."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.orchestrator import GenStreamErrorRaised  # noqa: E402
from routes.inference import _friendly_gen_stream_error  # noqa: E402


@pytest.fixture
def logged(monkeypatch):
    lines = []
    monkeypatch.setattr(
        "routes.inference.logger",
        SimpleNamespace(error = lambda fmt, *args: lines.append(fmt % args)),
    )
    return lines


def test_a_flattened_failure_leaves_the_cause_behind(logged):
    cause = "Error: [METAL] Command buffer execution failed: Caused GPU Timeout Error."
    assert _friendly_gen_stream_error(GenStreamErrorRaised(cause)) == (
        "The GPU stopped responding. Reload the model to recover."
    )
    assert logged == [f"Local generation failed: {cause}"]


def test_an_unrecognised_failure_is_logged_even_though_the_client_learns_nothing(logged):
    error = GenStreamErrorRaised("Error: /srv/weights blew up")
    assert _friendly_gen_stream_error(error) == "An internal error occurred."
    assert logged == ["Local generation failed: Error: /srv/weights blew up"]


def test_a_public_message_is_returned_as_written(logged):
    error = GenStreamErrorRaised("Error: audio generation is in progress", public = True)
    assert _friendly_gen_stream_error(error) == "Error: audio generation is in progress"
