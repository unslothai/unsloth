# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend capture wiring for the self-note feature (Task 7.5).

`core/inference/self_note.py` ships the parsing helpers, but nothing outside their own
tests called them: the instruction was never injected into a prompt, a note the model
wrote was never parsed out, and the raw `<remember>` tag was never stripped from what the
user sees. This module tests the three wiring points added in `routes/inference.py`:

- `_apply_self_note_nudge` -- injects `self_note.note_instruction()` into the assembled
  system-prompt nudge, mirroring `_apply_compaction_nudge`.
- `_extract_and_strip_self_note` -- pulls the note out of a finished reply and returns the
  user-visible text with the `<remember>` block removed.
- `_model_json_response_with_self_note` -- attaches the note as a `self_note` content part
  on the JSON response's assistant message, the shape the client round-trips.

The hard requirement is that all of this is a no-op when `SELF_NOTE_ENABLED` is False, so
disabled behaviour stays byte-identical to before this task.
"""

from __future__ import annotations

from core.inference import self_note as self_note_module
from routes.inference import (
    _apply_self_note_nudge,
    _extract_and_strip_self_note,
    _model_json_response_with_self_note,
)


# ── 1. Instruction injection ──────────────────────────────────────────────


def test_enabled_nudge_carries_the_instruction(monkeypatch):
    monkeypatch.setattr(self_note_module, "SELF_NOTE_ENABLED", True)
    result = _apply_self_note_nudge("")
    assert self_note_module.SELF_NOTE_INSTRUCTION in result


def test_enabled_nudge_appends_to_existing_text(monkeypatch):
    monkeypatch.setattr(self_note_module, "SELF_NOTE_ENABLED", True)
    result = _apply_self_note_nudge("Existing nudge text.")
    assert result.startswith("Existing nudge text. ")
    assert self_note_module.SELF_NOTE_INSTRUCTION in result


def test_disabled_nudge_is_byte_identical(monkeypatch):
    monkeypatch.setattr(self_note_module, "SELF_NOTE_ENABLED", False)
    assert _apply_self_note_nudge("") == ""
    assert _apply_self_note_nudge("Existing nudge text.") == "Existing nudge text."


# ── 2 & 3. Parse the note out, strip it from visible text ─────────────────


def test_note_is_extracted_and_visible_text_is_stripped(monkeypatch):
    monkeypatch.setattr(self_note_module, "SELF_NOTE_ENABLED", True)
    text = "Here is my answer.\n<remember>note body</remember>"
    visible, note = _extract_and_strip_self_note(text)
    assert note == "note body"
    assert "<remember>" not in visible
    assert "note body" not in visible


def test_no_remember_block_yields_no_note(monkeypatch):
    monkeypatch.setattr(self_note_module, "SELF_NOTE_ENABLED", True)
    text = "An ordinary reply with nothing to carry."
    visible, note = _extract_and_strip_self_note(text)
    assert note == ""
    assert visible == text


def test_disabled_feature_never_extracts_or_strips(monkeypatch):
    monkeypatch.setattr(self_note_module, "SELF_NOTE_ENABLED", False)
    text = "Here is my answer.\n<remember>note body</remember>"
    visible, note = _extract_and_strip_self_note(text)
    # Documented behaviour: disabled means this helper is a pure no-op. A raw
    # <remember> block in the reply passes straight through unchanged (the model was
    # never told the tag exists, so this is an adversarial/legacy-history case, not the
    # normal path) rather than being silently parsed and hidden.
    assert note == ""
    assert visible == text


def test_malformed_unterminated_block_does_not_raise_and_yields_no_note(monkeypatch):
    monkeypatch.setattr(self_note_module, "SELF_NOTE_ENABLED", True)
    text = "An answer.\n<remember>cut off mid-note"
    visible, note = _extract_and_strip_self_note(text)
    assert note == ""
    assert visible == text


def test_non_string_input_degrades_to_no_note(monkeypatch):
    monkeypatch.setattr(self_note_module, "SELF_NOTE_ENABLED", True)
    visible, note = _extract_and_strip_self_note(None)  # type: ignore[arg-type]
    assert note == ""
    assert visible is None


# ── Response wiring: the self_note content part ────────────────────────────


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeModel:
    """Minimal stand-in for a pydantic ChatCompletion: model_dump()/model_dump_json()."""

    def __init__(self, content):
        self._content = content

    def _as_dict(self):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self._content},
                    "finish_reason": "stop",
                }
            ],
        }

    def model_dump(self):
        return self._as_dict()

    def model_dump_json(self):
        import json

        return json.dumps(self._as_dict())


def test_response_with_note_carries_a_self_note_content_part():
    model = _FakeModel("Visible answer.")
    response = _model_json_response_with_self_note(model, "note body")
    import json

    body = json.loads(bytes(response.body))
    parts = body["choices"][0]["message"]["content"]
    assert isinstance(parts, list)
    types = {p["type"] for p in parts}
    assert "self_note" in types
    assert "text" in types
    note_part = next(p for p in parts if p["type"] == "self_note")
    assert note_part["content"] == "note body"
    text_part = next(p for p in parts if p["type"] == "text")
    assert text_part["text"] == "Visible answer."


def test_response_without_note_is_unchanged_shape():
    model = _FakeModel("Visible answer.")
    response = _model_json_response_with_self_note(model, "")
    import json

    body = json.loads(bytes(response.body))
    # No note -> falls back to the plain serialization path: content stays a bare
    # string, not a content-part list. This is the byte-identical-when-disabled shape.
    assert body["choices"][0]["message"]["content"] == "Visible answer."
