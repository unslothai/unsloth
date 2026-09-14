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
    _SelfNoteStreamExtractor,
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


# ── Task 7.6: the GGUF SSE streaming path ──────────────────────────────────
#
# The Studio chat UI always streams (chat-adapter.ts sends stream: true), so the
# non-streaming wiring above is never exercised by a real user. `_SelfNoteStreamExtractor`
# is the class actually wired into the GGUF tool-loop SSE generator in
# `produce_openai_chat_completions`: it holds back a `<remember>` block (and any fragment
# of its tags, across token boundaries) from the visible deltas the client receives, the
# same holdback technique `_ResponsesReasoningExtractor` already uses for `<think>`.
#
# These tests drive the extractor directly with token-sized chunks -- exactly how the SSE
# loop feeds it one `visible_delta` at a time -- rather than the full streaming route,
# which is entangled with admission/monitor/cancellation machinery not worth mocking here.


def _stream_tokens(extractor: _SelfNoteStreamExtractor, tokens: list[str]) -> str:
    """Feed ``tokens`` one at a time, as the SSE loop feeds one delta at a time.

    Returns the concatenation of everything the extractor allowed through -- i.e.
    everything that would have been yielded to the client as a visible delta.
    """
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
    # The extractor only decides what is SAFE TO SHOW; the actual capture, exactly as
    # wired in routes.inference, runs `extract_note` on the raw accumulated text (not the
    # filtered visible text) once the stream ends.
    tokens = ["Here is the answer.", "<remember>", "note body", "</remember>"]
    full_text = "".join(tokens)
    note = self_note_module.extract_note(full_text)
    assert note == "note body"


def test_tag_split_across_token_boundaries_is_held_back_and_still_captured():
    # The case naive per-token stripping misses: neither "<reme" nor "mber>" nor
    # "</rem" nor "ember>" contains the whole tag on its own, so a token-by-token
    # search-and-remove would let fragments (or the whole raw tag) leak through.
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
    # Mirrors the exact gate in the SSE loop: `_self_note_extractor` is only constructed
    # when the feature is enabled, and `shown` falls back to the raw delta unchanged.
    tokens = ["Here is the answer.", "<remember>", "note body", "</remember>"]
    extractor = _SelfNoteStreamExtractor() if self_note_module.enabled() else None
    shown_parts = []
    for tok in tokens:
        shown_parts.append(extractor.feed(tok) if extractor is not None else tok)
    shown = "".join(shown_parts)
    # Byte-identical to the raw stream: nothing held back, nothing stripped.
    assert shown == "".join(tokens)
    assert extractor is None


def test_unterminated_remember_does_not_hang_or_raise_and_drops_only_the_note():
    # Generation cut off mid-note (e.g. hit max_tokens before the closing tag).
    extractor = _SelfNoteStreamExtractor()
    tokens = ["Visible answer that finished. ", "<remember>", "cut off mid-note"]
    shown = _stream_tokens(extractor, tokens)
    # finish() must not raise and must not hang (there is no I/O in it, but it must
    # return promptly and deterministically at true stream end).
    extractor.finish()
    # The visible answer before the tag is untouched; the unterminated note (and its
    # partial tag) is held back and then dropped, not flushed as visible text -- flushing
    # it would leak a half-written note into the reply, and `extract_note` already treats
    # an unterminated block as no note at all, so this keeps both sides of the feature
    # consistent (see `_SelfNoteStreamExtractor.finish()`'s docstring for the reasoning).
    assert shown == "Visible answer that finished. "
    assert "remember" not in shown.lower()
    assert "cut off" not in shown

    full_text = "".join(tokens)
    note = self_note_module.extract_note(full_text)
    assert note == ""  # extract_note agrees: an unterminated block is no note.
