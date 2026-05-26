# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from utils.chat_template_validation import validate_chat_template


def test_chat_template_validation_accepts_plain_jinja():
    template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}"
        "{% endfor %}"
    )

    assert validate_chat_template(template) is None


def test_chat_template_validation_accepts_generation_blocks():
    template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] }}{% endgeneration %}"
        "{% endif %}"
        "{% endfor %}"
    )

    assert validate_chat_template(template) is None


def test_chat_template_validation_rejects_unclosed_generation_block():
    error = validate_chat_template("{% generation %}{{ content }}")

    assert error is not None
    assert "endgeneration" in error


def test_chat_template_validation_rejects_malformed_expression():
    error = validate_chat_template("{{ foo( }}")

    assert error is not None
    assert "expected ')'" in error
