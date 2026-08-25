# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Unit tests for the Anthropic-compatible thinking parameter.

Covers:
- ThinkingConfig model validation
- ChatCompletionRequest with thinking parameter
- Mapping logic: thinking.type -> enable_thinking
"""

import os
import sys

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from models.inference import ChatCompletionRequest, ThinkingConfig


def test_thinking_config_defaults_to_disabled():
    """ThinkingConfig should default to type='disabled'."""
    config = ThinkingConfig()
    assert config.type == "disabled"


def test_thinking_config_explicit_disabled():
    """ThinkingConfig should accept type='disabled'."""
    config = ThinkingConfig(type = "disabled")
    assert config.type == "disabled"


def test_thinking_config_explicit_enabled():
    """ThinkingConfig should accept type='enabled'."""
    config = ThinkingConfig(type = "enabled")
    assert config.type == "enabled"


def test_chat_completion_request_with_thinking_disabled():
    """thinking.type='disabled' should map to enable_thinking=False."""
    req = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "disabled"},
        }
    )
    assert req.thinking is not None
    assert req.thinking.type == "disabled"
    assert req.enable_thinking is False


def test_chat_completion_request_with_thinking_enabled():
    """thinking.type='enabled' should map to enable_thinking=True."""
    req = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "enabled"},
        }
    )
    assert req.thinking is not None
    assert req.thinking.type == "enabled"
    assert req.enable_thinking is True
    assert "enable_thinking" not in req.model_fields_set


def test_chat_completion_request_without_thinking():
    """ChatCompletionRequest should work without thinking parameter."""
    req = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    assert req.thinking is None
    assert req.enable_thinking is None


def test_chat_completion_request_backward_compatible_enable_thinking():
    """ChatCompletionRequest should still support enable_thinking."""
    req = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "enable_thinking": True,
        }
    )
    assert req.enable_thinking is True
    assert req.thinking is None


def test_thinking_overrides_enable_thinking_when_both_provided():
    """When both thinking and enable_thinking are provided,
    enable_thinking takes precedence (no override)."""
    req = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "enabled"},
            "enable_thinking": False,
        }
    )
    # enable_thinking is explicitly set, so it takes precedence
    assert req.enable_thinking is False
    assert "enable_thinking" in req.model_fields_set
    assert req.thinking.type == "enabled"


def test_thinking_mapping_ignores_an_explicitly_null_enable_thinking():
    """A client that serializes every optional field must not change precedence.

    pydantic records an explicit ``null`` in ``model_fields_set``, so without the
    discard the derived value reads as a typed override and the route drops a
    higher-priority nested control that the omitted form honors.
    """
    req = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking": {"type": "disabled"},
            "enable_thinking": None,
        }
    )
    assert req.enable_thinking is False
    assert "enable_thinking" not in req.model_fields_set


def test_explicit_null_and_omitted_enable_thinking_resolve_the_same():
    from routes.inference import _normalize_chat_reasoning_controls

    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "thinking": {"type": "disabled"},
        "chat_template_kwargs": {"reasoning_effort": "high"},
    }
    with_null = ChatCompletionRequest.model_validate({**body, "enable_thinking": None})
    omitted = ChatCompletionRequest.model_validate(dict(body))
    for payload in (with_null, omitted):
        _normalize_chat_reasoning_controls(payload)

    assert (with_null.enable_thinking, with_null.reasoning_effort) == (True, "high")
    assert (omitted.enable_thinking, omitted.reasoning_effort) == (True, "high")
