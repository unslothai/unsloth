# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Gemma-4 GGUF loads auto-apply the bundled gemma-4.jinja template, so quants need no re-download."""

from __future__ import annotations

import importlib.util
import sys
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import pytest

# Load chat_templates directly to avoid core.inference import side effects.
_CT_PATH = Path(_BACKEND_DIR) / "core" / "inference" / "chat_templates.py"
_ct_spec = importlib.util.spec_from_file_location("_gemma4_ct_test", _CT_PATH)
chat_templates = importlib.util.module_from_spec(_ct_spec)
_ct_spec.loader.exec_module(chat_templates)

is_unsloth_gemma4_gguf = chat_templates.is_unsloth_gemma4_gguf
resolve_effective_chat_template_override = chat_templates.resolve_effective_chat_template_override
load_bundled_chat_template = chat_templates.load_bundled_chat_template
is_unsloth_gemma4_edge_gguf = chat_templates.is_unsloth_gemma4_edge_gguf

BUNDLED = load_bundled_chat_template("gemma-4.jinja")  # 12b / 26B-A4B / 31B
EDGE = load_bundled_chat_template("gemma-4-edge.jinja")  # E2B / E4B


# Stubs let core.inference.llama_cpp import without the full studio venv.
def _stub_modules_ctx():
    """Stubs only heavy deps that are not importable, and removes the stubs on exit."""
    from unittest.mock import patch

    _loggers_stub = _types.ModuleType("loggers")
    _loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
    _structlog_stub = _types.ModuleType("structlog")
    _structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    _httpx_stub = _types.ModuleType("httpx")
    for _exc in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
    ):
        setattr(_httpx_stub, _exc, type(_exc, (Exception,), {}))
    _httpx_stub.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
    _httpx_stub.Client = type(
        "C",
        (),
        {
            "__init__": lambda s, **kw: None,
            "__enter__": lambda s: s,
            "__exit__": lambda s, *a: None,
        },
    )
    overrides = {
        name: stub
        for name, stub in (
            ("loggers", _loggers_stub),
            ("structlog", _structlog_stub),
            ("httpx", _httpx_stub),
        )
        if name not in sys.modules
    }
    return patch.dict(sys.modules, overrides)


def _detect_reasoning_flags():
    with _stub_modules_ctx():
        from core.inference.llama_cpp import detect_reasoning_flags
    return detect_reasoning_flags


@pytest.mark.parametrize(
    "model_id,expected",
    [
        ("unsloth/gemma-4-E2B-it-GGUF", True),
        ("unsloth/gemma-4-E4B-it-GGUF", True),
        ("unsloth/gemma-4-31B-it-GGUF", True),
        ("unsloth/gemma-4-26B-A4B-it-GGUF", True),
        ("unsloth/gemma-4-E4B-it-qat-GGUF", True),
        ("unsloth/gemma-4-12B-it-qat-GGUF", True),
        ("UNSLOTH/GEMMA-4-E2B-IT-GGUF", True),
        ("gemma-4-E2B-it-GGUF", True),  # owner-less shorthand -> unsloth/
        ("gemma-4-31B-it-GGUF", True),
        ("unsloth/gemma-4-E2B-it", False),
        ("unsloth/gemma-3-4b-it-GGUF", False),
        ("google/gemma-4-31B-it-GGUF", False),
        ("unsloth/Qwen3.5-9B-MTP-GGUF", False),
        ("/home/user/models/gemma-4-E2B.Q4_K_M.gguf", False),  # local path
        ("/scan/models--unsloth--gemma-4-E2B-it-GGUF/snapshots/abc", True),  # repo snapshot
        ("C:\\scan\\models--google--gemma-4-31B-it-GGUF\\snapshots\\abc", False),
        ("", False),
        (None, False),
    ],
)
def test_is_unsloth_gemma4_gguf(model_id, expected):
    assert is_unsloth_gemma4_gguf(model_id) is expected


@pytest.mark.parametrize(
    "model_id,expected_edge",
    [
        ("unsloth/gemma-4-E2B-it-GGUF", True),
        ("unsloth/gemma-4-E4B-it-GGUF", True),
        ("UNSLOTH/GEMMA-4-E4B-IT-GGUF", True),
        ("unsloth/gemma-4-E2B-it-qat-GGUF", True),
        ("unsloth/gemma-4-E4B-it-qat-GGUF", True),
        ("unsloth/gemma-4-E4B-it-qat-mobile-GGUF", True),
        ("unsloth/gemma-4-E2B-it-qat-mobile-GGUF", True),
        ("unsloth/gemma-4-12b-it-GGUF", False),
        ("unsloth/gemma-4-12B-it-qat-GGUF", False),
        ("unsloth/gemma-4-26B-A4B-it-GGUF", False),
        ("unsloth/gemma-4-31B-it-GGUF", False),
        ("unsloth/gemma-3-4b-it-GGUF", False),
    ],
)
def test_is_unsloth_gemma4_edge_gguf(model_id, expected_edge):
    assert is_unsloth_gemma4_edge_gguf(model_id) is expected_edge


def test_resolver_returns_edge_template_for_e2b_e4b():
    for mid in ("unsloth/gemma-4-E2B-it-GGUF", "unsloth/gemma-4-E4B-it-GGUF"):
        out = resolve_effective_chat_template_override(model_identifier = mid, user_override = None)
        assert out == EDGE
        assert out != BUNDLED


def test_resolver_routes_qat_repos_by_family():
    # QAT repos add '-qat'; E2B/E4B QAT must still get the edge template, or thinking-off
    # gets an empty thought block.
    for mid in ("unsloth/gemma-4-E2B-it-qat-GGUF", "unsloth/gemma-4-E4B-it-qat-GGUF"):
        out = resolve_effective_chat_template_override(model_identifier = mid, user_override = None)
        assert out == EDGE
    out = resolve_effective_chat_template_override(
        model_identifier = "unsloth/gemma-4-12B-it-qat-GGUF", user_override = None
    )
    assert out == BUNDLED


def test_resolver_handles_owner_less_shorthand():
    # The resolver runs before ModelConfig.from_identifier adds unsloth/, so it must normalize too.
    assert (
        resolve_effective_chat_template_override(
            model_identifier = "gemma-4-E2B-it-GGUF", user_override = None
        )
        == EDGE
    )
    assert (
        resolve_effective_chat_template_override(
            model_identifier = "gemma-4-31B-it-GGUF", user_override = None
        )
        == BUNDLED
    )


def test_resolver_returns_standard_template_for_larger_models():
    for mid in (
        "unsloth/gemma-4-12b-it-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "unsloth/gemma-4-31B-it-GGUF",
    ):
        out = resolve_effective_chat_template_override(model_identifier = mid, user_override = None)
        assert out == BUNDLED


def test_resolver_user_override_wins():
    out = resolve_effective_chat_template_override(
        model_identifier = "unsloth/gemma-4-E2B-it-GGUF", user_override = "MY TEMPLATE"
    )
    assert out == "MY TEMPLATE"


def test_resolver_blank_override_falls_back_to_bundled():
    out = resolve_effective_chat_template_override(
        model_identifier = "unsloth/gemma-4-31B-it-GGUF", user_override = "   "
    )
    assert out == BUNDLED


def test_resolver_none_for_non_gemma():
    assert (
        resolve_effective_chat_template_override(
            model_identifier = "unsloth/Llama-3.2-1B-Instruct-GGUF", user_override = None
        )
        is None
    )


@pytest.mark.parametrize("tpl", [BUNDLED, EDGE])
def test_bundled_template_has_preserve_thinking_defaulted_off(tpl):
    assert "preserve_thinking" in tpl
    assert "preserve_thinking | default(false)" in tpl


@pytest.mark.parametrize("name", ["gemma-4.jinja", "gemma-4-edge.jinja"])
def test_bundled_templates_are_ascii(name):
    # Bundled templates stay ASCII so the temp file encodes on non-UTF-8 Windows locales.
    text = load_bundled_chat_template(name)
    non_ascii = sorted({c for c in text if ord(c) > 127})
    assert not non_ascii, f"{name} has non-ASCII chars: {non_ascii}"


@pytest.mark.parametrize("tpl", [BUNDLED, EDGE])
def test_detect_reasoning_flags_on_bundled_template(tpl):
    detect_reasoning_flags = _detect_reasoning_flags()
    flags = detect_reasoning_flags(tpl, "unsloth/gemma-4-E2B-it-GGUF")
    assert flags["supports_reasoning"] is True
    assert flags["reasoning_style"] == "enable_thinking"
    assert flags["reasoning_always_on"] is False
    # This flag drives the UI 'Preserve thinking' toggle.
    assert flags["supports_preserve_thinking"] is True
    assert flags["supports_tools"] is True


def test_edge_template_omits_empty_thought_block_on_thinking_off():
    """E2B/E4B must NOT emit the empty <|channel>thought<channel|> block when
    thinking is disabled; the larger-model template must. This is the only
    intended difference between the two bundled templates."""
    EMPTY = "<|channel>thought\n<channel|>"
    msgs = [{"role": "user", "content": "hi"}]
    edge_off = _render_with(EDGE, msgs, enable_thinking = False)
    std_off = _render_with(BUNDLED, msgs, enable_thinking = False)
    assert EMPTY not in edge_off, "edge (E2B/E4B) should not emit empty thought block"
    assert EMPTY in std_off, "standard (12b/26B/31B) should emit empty thought block"
    assert EMPTY not in _render_with(EDGE, msgs, enable_thinking = True)


def _render_with(tpl, messages, **kw):
    pytest.importorskip("jinja2")  # transitive via transformers
    from jinja2 import Environment, BaseLoader

    def raise_exception(msg):
        raise RuntimeError(msg)

    env = Environment(loader = BaseLoader())
    return env.from_string(tpl).render(
        messages = messages,
        bos_token = "<bos>",
        raise_exception = raise_exception,
        add_generation_prompt = True,
        **kw,
    )


def _render(messages, **kw):
    return _render_with(BUNDLED, messages, **kw)


def _convo_with_prior_tool_reasoning():
    return [
        {"role": "user", "content": "q1"},
        {
            "role": "assistant",
            "reasoning_content": "SECRET_THOUGHT",
            "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": {"x": 1}}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "42"},
        {"role": "user", "content": "q2"},
    ]


def test_preserve_thinking_off_omits_prior_reasoning():
    assert "SECRET_THOUGHT" not in _render(_convo_with_prior_tool_reasoning())


def test_preserve_thinking_on_keeps_prior_reasoning():
    assert "SECRET_THOUGHT" in _render(_convo_with_prior_tool_reasoning(), preserve_thinking = True)


@pytest.mark.parametrize("tpl", [BUNDLED, EDGE])
def test_current_tool_reasoning_survives_a_suppressed_call_result(tpl):
    messages = _convo_with_prior_tool_reasoning()[:-1]
    messages[2]["content"] += "\n\nThe duplicate call was not executed."
    rendered = _render_with(tpl, messages)
    assert rendered.index("SECRET_THOUGHT") < rendered.index("The duplicate call was not executed.")


@pytest.mark.parametrize("tpl", [BUNDLED, EDGE])
def test_model_thinks_again_after_a_tool_result(tpl):
    messages = _convo_with_prior_tool_reasoning()[:-1]
    assert _render_with(tpl, messages, enable_thinking = True).endswith(
        "<tool_response|><|channel>thought\n"
    )
    assert _render_with(tpl, messages, enable_thinking = False).endswith("<tool_response|>")


@pytest.mark.parametrize("tpl", [BUNDLED, EDGE])
def test_closed_tool_turn_reopens_the_model_turn(tpl):
    messages = _convo_with_prior_tool_reasoning()[:-1]
    messages[1]["content"] = "Let me check."
    assert _render_with(tpl, messages, enable_thinking = True).endswith(
        "Let me check.<turn|>\n<|turn>model\n"
    )


@pytest.mark.parametrize("tpl", [BUNDLED, EDGE])
def test_consecutive_assistant_content_keeps_its_separator(tpl):
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    assert "first\nsecond<turn|>" in _render_with(tpl, messages)


@pytest.mark.parametrize("tpl", [BUNDLED, EDGE])
def test_completed_tool_turn_continuation_keeps_its_separator(tpl):
    messages = _convo_with_prior_tool_reasoning()[:-1]
    messages[1]["content"] = "Let me check."
    messages.append({"role": "assistant", "content": "The answer is 42."})
    assert "Let me check.\nThe answer is 42.<turn|>" in _render_with(tpl, messages)


def test_enable_thinking_gates_think_token():
    assert "<|think|>" in _render([{"role": "user", "content": "hi"}], enable_thinking = True)
    assert "<|think|>" not in _render([{"role": "user", "content": "hi"}], enable_thinking = False)


def test_already_in_target_state_consistent_with_bundled_override():
    """the bundled template replaces None before dedup to prevent needless reloads"""
    LlamaCppBackend, GgufLoadIntent = _import_backend()

    class _FakeProcess:
        def terminate(self): ...
        def wait(self, timeout = None):
            return 0

        def kill(self): ...
        def poll(self):
            return 0

    backend = LlamaCppBackend()
    backend._process = _FakeProcess()
    backend._healthy = True
    backend._model_identifier = "unsloth/gemma-4-E2B-it-GGUF"
    backend._hf_variant = "Q4_K_M"
    backend._requested_n_ctx = 8192
    backend._cache_type_kv = None
    backend._speculative_type = None
    backend._requested_spec_mode = "auto"
    backend._chat_template_override = BUNDLED
    backend._is_vision = False
    backend._extra_args = None
    backend._gguf_path = None

    common = dict(
        model_identifier = "unsloth/gemma-4-E2B-it-GGUF",
        hf_variant = "Q4_K_M",
        n_ctx = 8192,
        cache_type_kv = None,
        speculative_type = None,
        extra_args = None,
        is_vision = False,
    )
    assert backend.adopt_load_intent_if_matched(
        GgufLoadIntent(chat_template_override = BUNDLED, **common)
    )
    # Raw None would mismatch and force a needless reload.
    assert not backend.adopt_load_intent_if_matched(
        GgufLoadIntent(chat_template_override = None, **common)
    )


def _import_backend():
    with _stub_modules_ctx():
        from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend
    return LlamaCppBackend, GgufLoadIntent
