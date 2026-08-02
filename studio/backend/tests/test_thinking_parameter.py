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
    assert req.thinking.type == "enabled"
