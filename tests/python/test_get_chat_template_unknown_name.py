# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""An unknown or empty chat_template name must name the problem, not leak KeyError/IndexError."""

import pytest

from unsloth.chat_templates import CHAT_TEMPLATES, get_chat_template


class _FakeTokenizer:
    """Minimal surface get_chat_template touches before the template lookup."""

    is_fast = True
    padding_side = "right"


def test_unknown_template_name_lists_the_supported_names():
    with pytest.raises(ValueError) as exc_info:
        get_chat_template(_FakeTokenizer(), chat_template = "not-a-template")

    message = str(exc_info.value)
    assert "not-a-template" in message
    assert "alpaca" in message
    assert "chatml" in message


def test_empty_template_list_raises_a_clear_error():
    with pytest.raises(ValueError) as exc_info:
        get_chat_template(_FakeTokenizer(), chat_template = [])

    assert "empty list" in str(exc_info.value)


def test_known_template_names_still_resolve():
    assert "chatml" in CHAT_TEMPLATES
