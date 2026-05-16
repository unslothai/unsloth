# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the orphan tool_call_id pairing helper.

PR 5375 had ChatMessage._validate_role_shape synthesise a random
``tool_call_id`` when ``role="tool"`` arrived without one (frontend's
second-round POST drops the streamed id). The random id broke
correlation with the preceding assistant ``tool_calls`` ids -- upstream
OpenAI-compatible backends reject "tool result not referenced by any
tool_call". The follow-up fix:

* The validator tags the synthesised id with a recognisable prefix
  (``TOOL_CALL_ID_SYNTH_PREFIX``).
* The route handler runs ``_pair_orphan_tool_ids`` before passthrough
  to rewrite synth ids to the matching announced assistant tool_call.

This module pins the rewrite contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from models.inference import TOOL_CALL_ID_SYNTH_PREFIX
from routes.inference import _pair_orphan_tool_ids


def _synth(idx: int = 0) -> str:
    return f"{TOOL_CALL_ID_SYNTH_PREFIX}aa{idx:02d}"


class TestPairing:
    def test_synth_id_rewritten_to_preceding_assistant(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_abc123", "type": "function"}],
            },
            {"role": "tool", "tool_call_id": _synth(0), "content": "result"},
        ]
        out = _pair_orphan_tool_ids(msgs)
        assert out[-1]["tool_call_id"] == "call_abc123"
        # Other messages untouched and not aliased.
        assert out[0] is msgs[0]
        assert out[1] is msgs[1]
        assert out[2] is not msgs[2]

    def test_real_id_left_alone(self):
        real = "call_real_001"
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [{"id": real, "type": "function"}],
            },
            {"role": "tool", "tool_call_id": real, "content": "ok"},
        ]
        out = _pair_orphan_tool_ids(msgs)
        # Idempotent: nothing rewritten.
        assert out == msgs

    def test_multiple_synths_pair_to_distinct_calls(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_a", "type": "function"},
                    {"id": "call_b", "type": "function"},
                ],
            },
            {"role": "tool", "tool_call_id": _synth(0), "content": "x"},
            {"role": "tool", "tool_call_id": _synth(1), "content": "y"},
        ]
        out = _pair_orphan_tool_ids(msgs)
        assert out[1]["tool_call_id"] == "call_a"
        assert out[2]["tool_call_id"] == "call_b"

    def test_synth_left_alone_when_no_announced_call(self):
        # No preceding assistant tool_calls; the synth id stays so the
        # upstream backend can produce a clear error.
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "tool", "tool_call_id": _synth(0), "content": "x"},
        ]
        out = _pair_orphan_tool_ids(msgs)
        assert out[-1]["tool_call_id"].startswith(TOOL_CALL_ID_SYNTH_PREFIX)

    def test_existing_real_call_not_double_consumed(self):
        msgs = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "call_a", "type": "function"},
                    {"id": "call_b", "type": "function"},
                ],
            },
            # First tool result already references call_a explicitly.
            {"role": "tool", "tool_call_id": "call_a", "content": "x"},
            # Second is a synth; should map to the remaining call_b.
            {"role": "tool", "tool_call_id": _synth(0), "content": "y"},
        ]
        out = _pair_orphan_tool_ids(msgs)
        assert out[1]["tool_call_id"] == "call_a"
        assert out[2]["tool_call_id"] == "call_b"

    def test_no_synths_returns_original(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        out = _pair_orphan_tool_ids(msgs)
        # No rewrites needed -- return the same list reference.
        assert out is msgs


class TestValidatorSynth:
    """Mirror the validator's behaviour so the synth prefix stays stable."""

    def test_validator_emits_synth_prefix(self):
        from models.inference import ChatMessage

        m = ChatMessage(role = "tool", content = "result")
        assert m.tool_call_id is not None
        assert m.tool_call_id.startswith(TOOL_CALL_ID_SYNTH_PREFIX)

    def test_validator_keeps_explicit_id(self):
        from models.inference import ChatMessage

        m = ChatMessage(role = "tool", tool_call_id = "call_real_xyz", content = "ok")
        assert m.tool_call_id == "call_real_xyz"
