# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import _note_before_envelopes
from core.inference.mcp_images import (
    SENTINEL,
    UNPARSED_IMAGES_TEXT,
    promote_history,
    split_images,
)
from core.inference.tool_call_parser import (
    repeated_result_message,
    starved_result_message,
)
from core.inference.tool_loop_controller import strip_result_for_model

TOOL = "mcp__a5d934d4713a4f20__screenshot"
BASE64 = "iVBORw0KGgo" + "A" * 200_000
# What _fit_result_to_room hands back for an MCP screenshot: a cut body, envelope kept whole.
FITTED = (
    "page text\n\n... (truncated to 10 chars for the model; 99 chars total.)\n[1 image returned]"
    f"\n{SENTINEL}{json.dumps([{'data': BASE64, 'mimeType': 'image/png'}])}"
)
# The saved shape from #11358: the notice landed after the array.
CORRUPTED = starved_result_message(TOOL, FITTED)


def test_appended_notice_is_what_broke_the_envelope():
    assert split_images(CORRUPTED)[1] == []


def test_starved_and_repeated_notes_keep_the_envelope_parseable():
    for add, note in (
        (lambda body: starved_result_message(TOOL, body), "No room in the window"),
        (lambda body: repeated_result_message(TOOL, 3, body), "returned exactly this 3 times"),
    ):
        result = _note_before_envelopes(FITTED, TOOL, add)
        text, images = split_images(result)
        assert images and images[0]["data"] == BASE64
        assert text.startswith("page text") and note in text
        model_text = strip_result_for_model(result, TOOL)
        assert BASE64[:40] not in model_text
        assert "[1 image returned]" in model_text


def test_saved_corrupted_result_is_cut_for_the_model():
    model_text = strip_result_for_model(CORRUPTED, TOOL)
    assert BASE64[:40] not in model_text
    assert model_text.startswith("page text")
    assert model_text.endswith(UNPARSED_IMAGES_TEXT)


def test_saved_corrupted_result_is_cut_on_replay():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": TOOL, "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": CORRUPTED},
    ]
    for vision in (False, True):
        out = promote_history(messages, vision = vision)
        assert BASE64[:40] not in json.dumps(out)
        assert out[1]["content"].endswith(UNPARSED_IMAGES_TEXT)


def test_non_image_tools_and_prose_mentions_are_untouched():
    assert strip_result_for_model(CORRUPTED, "python") == CORRUPTED
    prose = f"docs say\n{SENTINEL} is the marker\nmore"
    assert strip_result_for_model(prose, TOOL) == prose
