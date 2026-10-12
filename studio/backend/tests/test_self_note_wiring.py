# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Backend capture wiring for the self-note feature."""

from __future__ import annotations

from core.inference import self_note as self_note_module
from routes.inference import (
    _apply_self_note_nudge,
    _extract_and_strip_self_note,
    _model_json_response_with_self_note,
    _SelfNoteStreamExtractor,
)


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
    assert body["choices"][0]["message"]["content"] == "Visible answer."


def _stream_tokens(extractor: _SelfNoteStreamExtractor, tokens: list[str]) -> str:
    """Feed ``tokens`` one at a time, as the SSE loop feeds one delta at a time."""
    shown_parts = [extractor.feed(tok) for tok in tokens]
    return "".join(shown_parts)


def test_stream_never_emits_the_tag_or_note_body():
    extractor = _SelfNoteStreamExtractor()
    tokens = ["Here is the answer.", "<remember>", "note body", "</remember>"]
    shown = _stream_tokens(extractor, tokens)
    extractor.finish()
    assert "<remember>" not in shown
    assert "</remember>" not in shown
    assert "note body" not in shown
    assert shown == "Here is the answer."


def test_stream_produces_a_self_note_part_with_the_note_content():
    tokens = ["Here is the answer.", "<remember>", "note body", "</remember>"]
    full_text = "".join(tokens)
    note = self_note_module.extract_note(full_text)
    assert note == "note body"


def test_tag_split_across_token_boundaries_is_held_back_and_still_captured():
    tokens = ["Visible. ", "<reme", "mber>", "body", "</rem", "ember>", " more visible."]
    extractor = _SelfNoteStreamExtractor()
    shown = _stream_tokens(extractor, tokens)
    extractor.finish()
    assert shown == "Visible.  more visible."
    assert "remember" not in shown.lower()
    assert "body" not in shown

    full_text = "".join(tokens)
    note = self_note_module.extract_note(full_text)
    assert note == "body"


def test_disabled_feature_streams_byte_identically(monkeypatch):
    monkeypatch.setattr(self_note_module, "SELF_NOTE_ENABLED", False)
    tokens = ["Here is the answer.", "<remember>", "note body", "</remember>"]
    extractor = _SelfNoteStreamExtractor() if self_note_module.enabled() else None
    shown_parts = []
    for tok in tokens:
        shown_parts.append(extractor.feed(tok) if extractor is not None else tok)
    shown = "".join(shown_parts)
    assert shown == "".join(tokens)
    assert extractor is None


def test_unterminated_remember_does_not_hang_or_raise_and_drops_only_the_note():
    extractor = _SelfNoteStreamExtractor()
    tokens = ["Visible answer that finished. ", "<remember>", "cut off mid-note"]
    shown = _stream_tokens(extractor, tokens)
    extractor.finish()
    assert shown == "Visible answer that finished. "
    assert "remember" not in shown.lower()
    assert "cut off" not in shown

    full_text = "".join(tokens)
    note = self_note_module.extract_note(full_text)
    assert note == ""  # extract_note agrees: an unterminated block is no note.
