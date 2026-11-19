# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import math

import pytest

from core.inference import tools


@pytest.fixture
def measured_room(monkeypatch):
    monkeypatch.setattr(tools.tool_result_limit, "is_automatic", lambda: False, raising=False)
    context = tools._REQUEST_CONTEXT_TOKENS.set(8192)
    room = tools._REQUEST_RESULT_BUDGET.set(4000)
    monkeypatch.setattr(tools, "_tool_result_max_chars", lambda: 50000)
    counter = lambda text, budget=0: 20 + math.ceil(len(text) / 5)
    monkeypatch.setattr(tools, "_loaded_token_counter", lambda ctx: counter)
    yield counter
    tools._REQUEST_RESULT_BUDGET.reset(room)
    tools._REQUEST_CONTEXT_TOKENS.reset(context)


def test_complete_result_that_fits_reserved_tokens_is_not_cut_by_window_share(measured_room):
    text = "ordinary prose " * 1200 + "\nLAST RECORD: 32"
    assert measured_room(text) < 4000
    result = tools._truncate(text)
    assert result == text
    assert "LAST RECORD: 32" in result


def test_configured_character_limit_still_wins(measured_room, monkeypatch):
    monkeypatch.setattr(tools, "_tool_result_max_chars", lambda: 3000)
    result = tools._truncate("ordinary prose " * 1200)
    assert "chars for the model;" in result
    assert len(result.split("\n\n... (truncated", 1)[0]) <= 3000


def test_dense_output_must_fit_real_tokens(measured_room, monkeypatch):
    counter = lambda text, budget=0: 20 + math.ceil(len(text) * 1.7)
    monkeypatch.setattr(tools, "_loaded_token_counter", lambda ctx: counter)
    text = "漢字🦥é" * 4000
    length = tools._dense_char_limit(text, 50000)
    assert length < len(text)
    assert counter(text[:length]) <= 4000


def test_failed_measurement_falls_back_conservatively(measured_room, monkeypatch):
    def counter(text, budget=0):
        return 20 + len(text) if len(text) <= 64 else None
    monkeypatch.setattr(tools, "_loaded_token_counter", lambda ctx: counter)
    text = "ordinary prose " * 1200
    length = tools._dense_char_limit(text, 50000)
    assert length <= 8000
    assert length < len(text)


def test_unknown_tokenizer_keeps_the_old_window_cap(measured_room, monkeypatch):
    monkeypatch.setattr(tools, "_loaded_token_counter", lambda ctx: None)
    assert tools._tool_result_char_budget() <= int(8192 * 4 * tools._PAGE_CONTEXT_SHARE)
    assert "chars for the model;" in tools._truncate("ordinary prose " * 1200)


def test_automatic_uses_measured_room_above_the_fallback_cap(measured_room, monkeypatch):
    monkeypatch.setattr(tools, "_tool_result_max_chars", lambda: 16000)
    monkeypatch.setattr(tools.tool_result_limit, "is_automatic", lambda: True, raising=False)
    text = "ordinary prose " * 1200 + "\nLAST RECORD: 32"
    assert len(text) > 16000
    assert tools._truncate(text) == text


def test_automatic_without_a_counter_keeps_the_fallback(measured_room, monkeypatch):
    monkeypatch.setattr(tools, "_tool_result_max_chars", lambda: 16000)
    monkeypatch.setattr(tools.tool_result_limit, "is_automatic", lambda: True, raising=False)
    monkeypatch.setattr(tools, "_loaded_token_counter", lambda ctx: None)
    assert tools._tool_result_char_budget() <= 16000


def test_automatic_still_has_a_bounded_character_ceiling(measured_room, monkeypatch):
    monkeypatch.setattr(tools.tool_result_limit, "is_automatic", lambda: True, raising=False)
    assert tools._tool_result_char_budget() == tools.tool_result_limit.MAX_MAX_CHARS
