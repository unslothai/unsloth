# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import tools
from core.inference.llama_cpp import _is_window_notice, _note_before_envelopes
from core.inference.mcp_images import (
    SENTINEL,
    promote_history,
    split_images,
)
from core.inference.tool_call_parser import (
    repeated_result_message,
    starved_result_message,
)
from core.inference.tool_loop_controller import strip_result_for_model

TOOL = "mcp__a5d934d4713a4f20__screenshot"


def _png() -> str:
    from PIL import Image

    out = io.BytesIO()
    Image.frombytes("RGB", (128, 128), os.urandom(128 * 128 * 3)).save(out, "PNG")
    return base64.b64encode(out.getvalue()).decode()


BASE64 = _png()
FITTED = (
    "page text\n\n... (truncated to 10 chars for the model; 99 chars total.)\n[1 image returned]"
    f"\n{SENTINEL}{json.dumps([{'data': BASE64, 'mimeType': 'image/png'}])}"
)
# The saved shape from #11358: the notice landed after the array.
CORRUPTED = starved_result_message(TOOL, FITTED)


def test_the_fitter_leaves_the_shape_the_starved_branch_fires_on(monkeypatch):
    raw = "page text\n" + "x" * 50_000 + FITTED[FITTED.index("\n[1 image returned]") :]
    monkeypatch.setattr(tools, "_loaded_context_tokens", lambda: 8192)
    token = tools._REQUEST_RESULT_BUDGET.set(200)
    try:
        fitted = tools._fit_result_to_room(raw, TOOL)
    finally:
        tools._REQUEST_RESULT_BUDGET.reset(token)
    assert _is_window_notice(fitted)
    assert split_images(fitted)[1][0]["data"] == BASE64
    assert not starved_result_message(TOOL, fitted).endswith("}]")
    noted = _note_before_envelopes(fitted, TOOL, lambda body: starved_result_message(TOOL, body))
    assert split_images(noted)[1][0]["data"] == BASE64
    assert BASE64[:40] not in strip_result_for_model(noted, TOOL)


def test_starved_and_repeated_notes_keep_the_envelope_parseable():
    for add, note in (
        (lambda body: starved_result_message(TOOL, body), "No room in the window"),
        (lambda body: repeated_result_message(TOOL, 3, body), "returned exactly this 3 times"),
    ):
        result = _note_before_envelopes(FITTED, TOOL, add)
        assert result.endswith("}]")
        text, images = split_images(result)
        assert images and images[0]["data"] == BASE64
        assert text.startswith("page text") and note in text


def test_saved_results_with_notes_after_the_array_are_repaired():
    both = repeated_result_message(TOOL, 3, CORRUPTED)
    for saved in (CORRUPTED, both):
        text, images = split_images(saved)
        assert images and images[0]["data"] == BASE64
        assert text.startswith("page text") and "No room in the window" in text
        model_text = strip_result_for_model(saved, TOOL)
        assert BASE64[:40] not in model_text and "No room in the window" in model_text
    assert "returned exactly this 3 times" in split_images(both)[0]


def test_saved_result_is_repaired_on_replay():
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
    plain = promote_history(messages, vision = False)
    assert BASE64[:40] not in json.dumps(plain)
    assert "No room in the window" in plain[1]["content"]
    promoted = []
    promote_history(messages, vision = True, promoted_out = promoted)
    assert len(promoted) == 1


def test_other_text_after_an_envelope_is_left_alone():
    # A file an MCP tool read may quote an envelope; only the loop's own notes are moved.
    quoted = FITTED + "\n\nmore of the file\n[1] footnote"
    assert split_images(quoted) == (quoted, [])
    assert strip_result_for_model(quoted, TOOL) == quoted
    prose = f"docs say\n{SENTINEL} is the marker\n\n[No room in the window for this result, so x]"
    assert split_images(prose) == (prose, [])
