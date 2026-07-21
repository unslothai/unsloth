# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Capability advertisement contract: classifier honesty, worker->orchestrator
IPC hop, route-layer end-to-end. Pure helpers + fakes; no torch/transformers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))


# Qwen3 snippet covering tools, enable_thinking, preserve_thinking.
QWEN3_TEMPLATE = """
{%- if tools %}
  {{- '<|im_start|>system\\nFor each function call, return a json object'
      ' wrapped inside <tool_call></tool_call> tags.\\n' }}
  {%- for tool in tools %}
    {{- tool | tojson }}
  {%- endfor %}
{%- endif %}
{%- for message in messages %}
  {%- if message.role == 'tool' %}
    {{- '<|im_start|>tool\\n' + message.content + '<|im_end|>\\n' }}
  {%- endif %}
{%- endfor %}
{%- if enable_thinking is defined and enable_thinking %}
  {{- '<think>' }}
{%- endif %}
{%- if preserve_thinking %}
  {{- assistant.reasoning_content }}
{%- endif %}
"""


GPT_OSS_TEMPLATE = """
<|start|>system<|message|>You are gpt-oss.
reasoning_effort: {{ reasoning_effort }}
<|end|>
"""


# DeepSeek-V4-Flash: an enable_thinking on/off gate PLUS a reasoning_effort
# 'max' preamble. The shipped template only *branches* on 'max' ('high' renders
# identically to thinking-on-without-the-preamble), so the literal scan alone
# would surface only ['max']; the classifier adds 'high' for deepseek-v4 to
# expose the encoder's full none/high/max ladder.
DEEPSEEK_V4_TEMPLATE = (
    "{%- if not thinking is defined %}"
    "{%- if enable_thinking is defined %}{%- set thinking = enable_thinking %}"
    "{%- else %}{%- set thinking = false %}{%- endif %}{%- endif %}\n"
    "{%- if thinking and reasoning_effort == 'max' %}"
    "{{- 'Reasoning Effort: Absolute maximum' }}{%- endif %}\n"
    "{%- for message in messages %}{{- message.content }}{%- endfor %}"
)


PLAIN_TEMPLATE = """
{%- for message in messages %}
  {{- message.role + ': ' + message.content + '\\n' }}
{%- endfor %}
"""


# ── Tests: classifier honesty ────────────────────────────────────────


def test_detect_reasoning_flags_qwen3_supports_tools_and_reasoning():
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(QWEN3_TEMPLATE, "unsloth/Qwen3-0.6B")
    assert flags["supports_tools"] is True
    assert flags["supports_reasoning"] is True
    assert flags["reasoning_style"] == "enable_thinking"
    assert flags["supports_preserve_thinking"] is True
    assert flags["reasoning_always_on"] is False


def test_detect_reasoning_flags_plain_template_all_false():
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(PLAIN_TEMPLATE, "some/PlainChat")
    assert flags["supports_tools"] is False
    assert flags["supports_reasoning"] is False
    assert flags["supports_preserve_thinking"] is False
    assert flags["reasoning_always_on"] is False


def test_detect_reasoning_flags_none_template_returns_all_false():
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(None)
    assert flags["supports_tools"] is False
    assert flags["supports_reasoning"] is False
    assert flags["supports_preserve_thinking"] is False
    assert flags["reasoning_always_on"] is False
    assert flags["reasoning_style"] == "enable_thinking"


def test_detect_reasoning_flags_deepseek_v4_exposes_none_high_max():
    """DeepSeek-V4-Flash: enable_thinking gate + reasoning_effort 'max' preamble.
    Classified as the hybrid style with the full none/high/max ladder even
    though the template only branches on 'max'."""
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(
        DEEPSEEK_V4_TEMPLATE, "unsloth/DeepSeek-V4-Flash-GGUF"
    )
    assert flags["supports_reasoning"] is True
    assert flags["reasoning_style"] == "enable_thinking_effort"
    assert flags["reasoning_effort_levels"] == ["high", "max"]
    assert flags["reasoning_always_on"] is False


def test_detect_reasoning_flags_non_deepseek_v4_effort_only_max_not_injected():
    """The 'high' injection is scoped to deepseek-v4: a different model whose
    template only branches on 'max' keeps ['max'] (no phantom 'high')."""
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(DEEPSEEK_V4_TEMPLATE, "vendor/OtherHybrid-GGUF")
    assert flags["reasoning_style"] == "enable_thinking_effort"
    assert flags["reasoning_effort_levels"] == ["max"]


def test_detect_safetensors_features_passes_template_through_to_classifier():
    """Route wrapper forwards a real template to the inner classifier."""
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/Qwen3-0.6B")
    flags = _detect_safetensors_features(backend, QWEN3_TEMPLATE)
    assert flags["supports_tools"] is True
    assert flags["supports_reasoning"] is True


def test_detect_safetensors_features_none_template_returns_all_false():
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/Qwen3-0.6B")
    flags = _detect_safetensors_features(backend, None)
    assert flags == {
        "supports_reasoning": False,
        "reasoning_style": "enable_thinking",
        "reasoning_always_on": False,
        "reasoning_effort_levels": [],
        "supports_preserve_thinking": False,
        "supports_tools": False,
    }


def test_detect_safetensors_features_gptoss_disables_tools():
    """gpt-oss Harmony: tools off even if template marks it."""
    from routes.inference import _detect_safetensors_features

    backend = MagicMock()
    backend.active_model_name = "unsloth/gpt-oss-20b"
    backend._is_gpt_oss_model.return_value = True

    flags = _detect_safetensors_features(backend, QWEN3_TEMPLATE)
    assert flags["supports_reasoning"] is True
    assert flags["reasoning_style"] == "reasoning_effort"
    assert flags["supports_tools"] is False


# Llama-3 / Mistral / Gemma 4 tool-call formats are now parser-supported, so supports_tools=True
# must hold for all of them; only templates matching none of the five known markers are suppressed.

LLAMA3_TEMPLATE = """
{%- if tools %}
  {{- '<|start_header_id|>system<|end_header_id|>' }}
  {{- 'You have access to the following tools.' }}
  {%- for tool in tools %}
    {{- tool | tojson }}
  {%- endfor %}
{%- endif %}
{%- for message in messages %}
  {%- if message.role == 'tool' %}
    {{- '<|start_header_id|>ipython<|end_header_id|>' }}
    {{- '<|python_tag|>' }}
    {{- message.content }}
  {%- endif %}
{%- endfor %}
"""

MISTRAL_TEMPLATE = """
{%- if tools %}
  {%- for tool in tools %}
    {{- tool | tojson }}
  {%- endfor %}
{%- endif %}
{%- for message in messages %}
  {%- if message.role == 'tool' %}
    {{- '[TOOL_CALLS]' + message.content + '[/TOOL_CALLS]' }}
  {%- endif %}
{%- endfor %}
"""

GEMMA4_TEMPLATE = """
{%- if tools %}
  {{- 'Tools available. Emit calls as ' }}
  {{- '<|tool_call>call:NAME{key:<|"|>val<|"|>}<tool_call|>' }}
  {%- for tool in tools %}
    {{- tool | tojson }}
  {%- endfor %}
{%- endif %}
"""


def test_detect_safetensors_features_llama3_template_keeps_tools_on():
    """Llama-3 emits <|python_tag|>; parser now supports it."""
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/Llama-3.2-3B-Instruct")
    flags = _detect_safetensors_features(backend, LLAMA3_TEMPLATE)
    assert flags["supports_tools"] is True


def test_detect_safetensors_features_mistral_template_keeps_tools_on():
    """Mistral emits [TOOL_CALLS]name{json}, which the safetensors loop now parses
    (the shared bracket-tag parser). The gate must no longer suppress it, or the
    PR's Mistral tool support is unreachable through normal capability detection."""
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/mistral-7b-instruct-v0.3")
    flags = _detect_safetensors_features(backend, MISTRAL_TEMPLATE)
    assert flags["supports_tools"] is True


def test_detect_safetensors_features_gemma4_template_keeps_tools_on():
    """Gemma 4 emits <|tool_call>; parser now supports it."""
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/gemma-4-E2B-it-UD-MLX-4bit")
    flags = _detect_safetensors_features(backend, GEMMA4_TEMPLATE)
    assert flags["supports_tools"] is True


# DeepSeek V3 / V3.1 / R1 emit ``<｜tool▁calls▁begin｜>...`` blocks.
# Note the full-width pipe (U+FF5C) and lower-1/8-block (U+2581).
DEEPSEEK_TEMPLATE = """
{%- if tools %}
  {%- for tool in tools %}
    {{- tool | tojson }}
  {%- endfor %}
{%- endif %}
{%- for message in messages %}
  {%- if message.role == 'assistant' and message.tool_calls %}
    {%- for tc in message.tool_calls %}
      {{- '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tc.function.name +
          '<｜tool▁sep｜>' + tc.function.arguments + '<｜tool▁call▁end｜>' }}
    {%- endfor %}
  {%- endif %}
{%- endfor %}
"""


def test_detect_safetensors_features_deepseek_template_keeps_tools_on():
    """DeepSeek emits ``<｜tool▁calls▁begin｜>...``; parser now supports it."""
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/DeepSeek-V3.1")
    flags = _detect_safetensors_features(backend, DEEPSEEK_TEMPLATE)
    assert flags["supports_tools"] is True


# GLM 4.5 / 4.6 / 4.7 emit ``<tool_call>NAME\n<arg_key>...<arg_value>...
GLM_TEMPLATE = """
{%- if tools %}
  For each function call, output the function name and arguments within
  the following XML format:
  <tool_call>{function-name}
  <arg_key>{arg-key}</arg_key>
  <arg_value>{arg-value}</arg_value>
  </tool_call>
  {%- for tool in tools %}
    {{- tool | tojson }}
  {%- endfor %}
{%- endif %}
"""


def test_detect_safetensors_features_glm_template_keeps_tools_on():
    """GLM 4.x emits ``<tool_call>NAME\\n<arg_key>...``; parser handles it."""
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/GLM-4.6")
    flags = _detect_safetensors_features(backend, GLM_TEMPLATE)
    assert flags["supports_tools"] is True


# Kimi K2 / Moonshot uses ``<|tool_calls_section_begin|>...`` blocks
# with ``functions.NAME:IDX`` as the per-call id.
KIMI_TEMPLATE = """
{%- if tools %}
  <|im_system|>tool_declare<|im_middle|>{{ tools | tojson }}<|im_end|>
{%- endif %}
{%- for message in messages %}
  {%- if message.role == 'assistant' and message.tool_calls %}
    <|tool_calls_section_begin|>
    {%- for tc in message.tool_calls %}
      <|tool_call_begin|>{{ tc.id }}<|tool_call_argument_begin|>{{ tc.function.arguments | tojson }}<|tool_call_end|>
    {%- endfor %}
    <|tool_calls_section_end|>
  {%- endif %}
{%- endfor %}
"""


def test_detect_safetensors_features_kimi_template_keeps_tools_on():
    """Kimi K2 emits ``<|tool_calls_section_begin|>...``; parser handles it."""
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/Kimi-K2-Instruct")
    flags = _detect_safetensors_features(backend, KIMI_TEMPLATE)
    assert flags["supports_tools"] is True


LLAMA3_2_BARE_JSON_TEMPLATE = """
{%- if tools %}
  {{- 'Given the following functions, respond with JSON for a function call.' }}
  {{- 'Respond in the format {"name": function name, "parameters": dictionary}.' }}
  {%- for tool in tools %}
    {{- tool | tojson }}
  {%- endfor %}
{%- endif %}
{%- for message in messages %}
  {%- if 'tool_calls' in message %}
    {{- '{"name": "' + message.tool_calls[0].function.name + '", '}}
    {{- '"parameters": ' + (message.tool_calls[0].function.arguments | tojson) + '}' }}
  {%- endif %}
{%- endfor %}
"""


def test_detect_safetensors_features_llama3_2_bare_json_keeps_tools_on():
    """Llama-3.2 bare JSON is supported, so the pill stays enabled."""
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/Llama-3.2-3B-Instruct")
    flags = _detect_safetensors_features(backend, LLAMA3_2_BARE_JSON_TEMPLATE)
    assert flags["supports_tools"] is True


MINICPM5_ATTRIBUTE_TEMPLATE = """
{%- if tools %}
  {{- 'Available tools. Emit calls as ' }}
  {{- '<function name="NAME"><parameter name="key">value</parameter></function>' }}
  {%- for tool in tools %}
    {{- tool | tojson }}
  {%- endfor %}
{%- endif %}
"""


def test_detect_safetensors_features_attribute_function_form_keeps_tools_on():
    """The attribute form ``<function name="...">`` must be whitelisted or the pill is wrongly suppressed."""
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "openbmb/MiniCPM-5")
    flags = _detect_safetensors_features(backend, MINICPM5_ATTRIBUTE_TEMPLATE)
    assert flags["supports_tools"] is True


def test_detect_safetensors_features_unknown_format_suppresses_tools():
    """Tools advertised with no known marker must be suppressed."""
    from routes.inference import _detect_safetensors_features

    tpl = (
        "{%- if tools %}<|im_start|>system\n"
        "Emit tool calls as JSON-RPC notifications inside the response."
        "<|im_end|>{%- endif %}"
    )
    backend = SimpleNamespace(active_model_name = "custom/unknown-tool-format")
    flags = _detect_safetensors_features(backend, tpl)
    assert flags["supports_tools"] is False


def test_detect_safetensors_features_qwen_tool_call_keeps_tools_on():
    """Sanity check: Qwen <tool_call> marker still flips supports_tools."""
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/Qwen3-0.6B")
    flags = _detect_safetensors_features(backend, QWEN3_TEMPLATE)
    assert flags["supports_tools"] is True


def test_detect_safetensors_features_function_xml_format_keeps_tools_on():
    """Templates emitting <function=name> XML are parser-compatible."""
    from routes.inference import _detect_safetensors_features

    tpl_with_function_xml = (
        "{%- if tools %}<|im_start|>system\n"
        "Tool call format: <function=name><parameter=k>v</parameter></function>"
        "<|im_end|>{%- endif %}"
    )
    backend = SimpleNamespace(active_model_name = "custom/with-function-xml")
    flags = _detect_safetensors_features(backend, tpl_with_function_xml)
    assert flags["supports_tools"] is True


def test_detect_safetensors_features_gemma_native_tool_call_keeps_tools_on():
    """Gemma 4 emits <|tool_call>call:name{...}<tool_call|>, which the shared
    parser now reads, so the gate must not suppress tools for it."""
    from routes.inference import _detect_safetensors_features

    tpl_with_gemma_native = (
        "{%- if tools -%}Tool call format: "
        "<|tool_call>call:name{key:value}<tool_call|>{%- endif -%}"
    )
    backend = SimpleNamespace(active_model_name = "unsloth/gemma-4-12b-it")
    flags = _detect_safetensors_features(backend, tpl_with_gemma_native)
    assert flags["supports_tools"] is True


def test_detect_safetensors_features_gemma_native_reasoning_is_parseable_not_prefilled():
    """Native Gemma channels are normalized to <think>, then split by the route."""
    from routes.inference import (
        _detect_safetensors_features,
        _sf_reasoning_prefill_mode,
    )

    tpl_with_gemma_native = (
        "{% if add_generation_prompt %}<|channel>thought\n<channel|>{% endif %}"
    )
    backend = SimpleNamespace(
        active_model_name = "unsloth/gemma-4-E2B-it",
        models = {
            "unsloth/gemma-4-E2B-it": {
                "native_chat_template": tpl_with_gemma_native,
                "chat_template_info": {"template": "override has no native markers"},
            }
        },
    )
    flags = _detect_safetensors_features(backend, "override has no native markers")
    missing_arg_flags = _detect_safetensors_features(backend, None)

    assert flags["supports_reasoning"] is True
    assert flags["reasoning_always_on"] is True
    assert missing_arg_flags["supports_reasoning"] is True
    assert _sf_reasoning_prefill_mode(flags, None, tpl_with_gemma_native) is False


def test_detect_safetensors_features_selects_native_reasoning_from_tool_template():
    """Request tools select a marker-bearing named template without affecting default chat."""
    from routes.inference import _detect_safetensors_features

    named_template = {
        "default": "plain default template",
        "tool_use": "{% if tools %}<|channel>thought\n<channel|>{% endif %}",
    }
    backend = SimpleNamespace(
        active_model_name = "custom/named-native-reasoning",
        models = {
            "custom/named-native-reasoning": {
                "native_chat_template": named_template,
                "chat_template_info": {
                    "template": "{% if tools %}<tool_call>{% endif %}"
                },
            }
        },
    )

    default_flags = _detect_safetensors_features(backend, "plain override")
    tool_flags = _detect_safetensors_features(
        backend,
        "plain override",
        tools = [{"type": "function"}],
    )

    assert default_flags["supports_reasoning"] is False
    assert tool_flags["supports_reasoning"] is True
    assert tool_flags["reasoning_always_on"] is True


# Qwen3.5 family pin: the live GGUF + safetensors templates both wrap tool
# calls as ``<tool_call>\n<function=name>...``. Faithful slice so the
# classifier never silently regresses for this family.

QWEN35_TOOL_INSTRUCTION = (
    "{%- if tools %}\n"
    "  <|im_start|>system\n"
    "  # Tools\n"
    "  <tools>\n"
    "  {%- for tool in tools %}{{ tool | tojson }}{%- endfor %}\n"
    "  </tools>\n"
    "  If you choose to call a function ONLY reply in the following format:\n"
    "  <tool_call>\n"
    "  <function=example_function_name>\n"
    "  <parameter=example_parameter_1>\n"
    "  value_1\n"
    "  </parameter>\n"
    "  </function>\n"
    "  </tool_call>\n"
    "  <|im_end|>\n"
    "{%- endif %}\n"
    "{%- if enable_thinking is defined and enable_thinking %}{{- '<think>' }}{%- endif %}\n"
)


def test_detect_safetensors_features_qwen35_keeps_tools_on():
    """unsloth/Qwen3.5-0.8B family must surface tools+reasoning on."""
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/Qwen3.5-0.8B")
    flags = _detect_safetensors_features(backend, QWEN35_TOOL_INSTRUCTION)
    assert flags["supports_tools"] is True
    assert flags["supports_reasoning"] is True
    assert flags["reasoning_style"] == "enable_thinking"


# ── Tests: IPC bridge contract ───────────────────────────────────────


def test_orchestrator_mirrors_chat_template_info_into_models_dict():
    """Worker → orchestrator copies chat_template_info verbatim."""
    from core.inference.orchestrator import InferenceOrchestrator

    orch = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orch.models = {}
    orch.active_model_name = None
    orch.loading_models = set()

    model_info = {
        "identifier": "unsloth/Qwen3-0.6B",
        "display_name": "Qwen3-0.6B",
        "is_vision": False,
        "is_lora": False,
        "is_gguf": False,
        "is_audio": False,
        "audio_type": None,
        "has_audio_input": False,
        "chat_template_info": {
            "has_template": True,
            "template": QWEN3_TEMPLATE,
            "format_type": "chatml",
            "template_name": "qwen3",
            "special_tokens": {"bos_token": "<|im_start|>"},
        },
    }

    # Replay orchestrator.load_model's mirror block.
    orch.active_model_name = model_info["identifier"]
    orch.models[orch.active_model_name] = {
        "is_vision": model_info.get("is_vision", False),
        "is_lora": model_info.get("is_lora", False),
        "display_name": model_info.get("display_name", "x"),
        "is_audio": model_info.get("is_audio", False),
        "audio_type": model_info.get("audio_type"),
        "has_audio_input": model_info.get("has_audio_input", False),
    }
    _tpl_info = model_info.get("chat_template_info")
    if isinstance(_tpl_info, dict):
        orch.models[orch.active_model_name]["chat_template_info"] = _tpl_info

    entry = orch.models[orch.active_model_name]
    tpl = entry.get("chat_template_info", {}).get("template")
    assert tpl == QWEN3_TEMPLATE

    from routes.inference import _detect_safetensors_features

    flags = _detect_safetensors_features(
        SimpleNamespace(active_model_name = orch.active_model_name), tpl
    )
    assert flags["supports_tools"] is True
    assert flags["supports_reasoning"] is True


def test_orchestrator_missing_chat_template_info_falls_back_to_all_false():
    """Old / malformed worker reply: no crash, all flags False."""
    from core.inference.orchestrator import InferenceOrchestrator
    from routes.inference import _detect_safetensors_features

    orch = InferenceOrchestrator.__new__(InferenceOrchestrator)
    orch.models = {}
    orch.active_model_name = "unsloth/Qwen3-0.6B"

    model_info = {
        "identifier": "unsloth/Qwen3-0.6B",
        "is_vision": False,
        "is_lora": False,
        # NB: no chat_template_info key
    }
    orch.models[orch.active_model_name] = {
        "is_vision": False,
        "is_lora": False,
    }
    _tpl_info = model_info.get("chat_template_info")
    if isinstance(_tpl_info, dict):
        orch.models[orch.active_model_name]["chat_template_info"] = _tpl_info

    entry = orch.models[orch.active_model_name]
    tpl = entry.get("chat_template_info", {}).get("template")
    assert tpl is None

    flags = _detect_safetensors_features(
        SimpleNamespace(active_model_name = orch.active_model_name), tpl
    )
    assert flags["supports_tools"] is False


def test_worker_load_reply_payload_includes_chat_template_info():
    """Worker IPC reply carries chat_template_info dict."""

    class _StubBackend:
        def __init__(self, identifier, template):
            self.active_model_name = identifier
            self.models = {
                identifier: {
                    "chat_template_info": {
                        "has_template": True,
                        "template": template,
                        "format_type": "chatml",
                        "template_name": "qwen3",
                        "special_tokens": {"bos_token": "<|im_start|>"},
                    }
                }
            }

    backend = _StubBackend("unsloth/Qwen3-0.6B", QWEN3_TEMPLATE)
    mc = SimpleNamespace(
        identifier = "unsloth/Qwen3-0.6B",
        display_name = "Qwen3-0.6B",
        is_vision = False,
        is_lora = False,
    )

    # Replay the worker's payload-build block.
    model_info = {
        "identifier": mc.identifier,
        "display_name": mc.display_name,
        "is_vision": mc.is_vision,
        "is_lora": mc.is_lora,
        "is_gguf": False,
    }
    _bm = getattr(backend, "models", {}) or {}
    _entry = (
        _bm.get(mc.identifier)
        or _bm.get(getattr(backend, "active_model_name", None))
        or {}
    )
    _tpl_info = _entry.get("chat_template_info")
    if isinstance(_tpl_info, dict):
        model_info["chat_template_info"] = {
            "has_template": bool(_tpl_info.get("has_template", False)),
            "template": _tpl_info.get("template"),
            "format_type": _tpl_info.get("format_type", "generic"),
            "template_name": _tpl_info.get("template_name"),
            "special_tokens": _tpl_info.get("special_tokens", {}) or {},
        }

    assert "chat_template_info" in model_info
    assert model_info["chat_template_info"]["template"] == QWEN3_TEMPLATE
    assert model_info["chat_template_info"]["has_template"] is True


def test_worker_load_reply_payload_survives_missing_template():
    """Tokenizer with no chat_template still yields a valid reply."""

    class _StubBackend:
        def __init__(self):
            self.active_model_name = "legacy/no-template"
            self.models = {"legacy/no-template": {}}  # no chat_template_info

    backend = _StubBackend()
    mc = SimpleNamespace(
        identifier = "legacy/no-template",
        display_name = "legacy",
        is_vision = False,
        is_lora = False,
    )

    model_info = {
        "identifier": mc.identifier,
        "display_name": mc.display_name,
        "is_vision": mc.is_vision,
        "is_lora": mc.is_lora,
        "is_gguf": False,
    }
    _bm = getattr(backend, "models", {}) or {}
    _entry = _bm.get(mc.identifier) or {}
    _tpl_info = _entry.get("chat_template_info")
    if isinstance(_tpl_info, dict):
        model_info["chat_template_info"] = dict(_tpl_info)

    assert "chat_template_info" not in model_info


# ── End-to-end: route layer sees the template, advertises True ───────


def test_route_layer_emits_supports_tools_true_for_qwen3_safetensors():
    """E2E: Qwen3 safetensors flips supports_tools=True."""
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(
        active_model_name = "unsloth/Qwen3-0.6B",
        models = {
            "unsloth/Qwen3-0.6B": {
                "is_vision": False,
                "chat_template_info": {
                    "has_template": True,
                    "template": QWEN3_TEMPLATE,
                    "format_type": "chatml",
                },
            }
        },
    )

    _model_info = backend.models.get(backend.active_model_name, {})
    _tpl = _model_info.get("chat_template_info", {}).get("template")
    flags = _detect_safetensors_features(backend, _tpl)

    assert flags["supports_tools"] is True
    assert flags["supports_reasoning"] is True
    assert flags["supports_preserve_thinking"] is True


@pytest.mark.parametrize(
    "opener",
    [
        "<｜tool▁calls▁begin｜>",  # canonical
        "<｜tool_calls_begin｜>",  # ASCII underscores
        "<｜tool▁calls｜>",  # short form
        "<｜tool calls begin｜>",  # spaces
        "<｜tool\\_calls\\_begin｜>",  # escaped underscores
    ],
)
def test_detect_safetensors_features_deepseek_opener_variants_keep_tools_on(opener):
    # Every DeepSeek opener the parser accepts must keep supports_tools on; the route gate derives
    # its markers from the parser's TOOL_XML_SIGNALS so it can no longer drift behind the parser ...
    from routes.inference import _detect_safetensors_features

    tpl = (
        "{%- if tools %}tools{%- endif %}"
        + opener
        + "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time{}"
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )
    backend = SimpleNamespace(active_model_name = "unsloth/DeepSeek-V3.1")
    flags = _detect_safetensors_features(backend, tpl)
    assert flags["supports_tools"] is True


# Templates that advertise tools ({%- if tools %}) and prompt the bare-JSON
# call form, but whose ``{"name":`` example is pretty-printed or JSON-escaped.
_WHITESPACE_BARE_JSON_TEMPLATE = (
    "{%- if tools %}\n"
    "To call a tool, output JSON of the form:\n"
    '{ "name" : "function_name", "parameters": { } }\n'
    "{%- endif %}\n"
    "{{ messages }}"
)
_ESCAPED_BARE_JSON_TEMPLATE = (
    "{%- if tools %}\n"
    'Respond with {\\"name\\": \\"fn\\", \\"parameters\\": {}}\n'
    "{%- endif %}\n"
    "{{ messages }}"
)
_TOOLS_ADVERTISED_NO_PARSEABLE_FORM = (
    "{%- if tools %}\nYou may use the available tools.\n{%- endif %}\n{{ messages }}"
)


def test_detect_safetensors_features_keeps_tools_for_pretty_printed_bare_json():
    # A pretty-printed bare-JSON example (``{ "name" :``) must keep supports_tools since the parser
    # accepts that whitespace via raw_decode.
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/Llama-3.2-3B-Instruct")
    flags = _detect_safetensors_features(backend, _WHITESPACE_BARE_JSON_TEMPLATE)
    assert flags["supports_tools"] is True


def test_detect_safetensors_features_keeps_tools_for_escaped_bare_json():
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/Llama-3.2-3B-Instruct")
    flags = _detect_safetensors_features(backend, _ESCAPED_BARE_JSON_TEMPLATE)
    assert flags["supports_tools"] is True


def test_detect_safetensors_features_drops_tools_when_no_parseable_form():
    # Negative control: tools advertised but no parser-recognised emission form at
    # all -> the pill is still dropped (the gate is not now matching everything).
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/Llama-3.2-3B-Instruct")
    flags = _detect_safetensors_features(backend, _TOOLS_ADVERTISED_NO_PARSEABLE_FORM)
    assert flags["supports_tools"] is False


def test_detect_safetensors_features_keeps_tools_for_function_alias_bare_json():
    # A template documenting the parser-supported {"function":...} bare-JSON alias
    # must keep supports_tools, mirroring the {"name":...} form.
    from routes.inference import _detect_safetensors_features

    tpl = (
        "{%- if tools %}\n"
        'Respond with {"function": "fn", "parameters": {}}\n'
        "{%- endif %}\n"
        "{{ messages }}"
    )
    backend = SimpleNamespace(active_model_name = "unsloth/Llama-3.2-3B-Instruct")
    flags = _detect_safetensors_features(backend, tpl)
    assert flags["supports_tools"] is True


# _sf_reasoning_prefill_mode gates the prefilled-<think> extractor (GGUF reasoning parity).
class TestSafetensorsReasoningPrefillGate:
    # A minimal Qwen3-style template with the standard <think>/</think> markers.
    _QWEN_TPL = "{% if enable_thinking %}<think>{% endif %}...</think>..."
    # gemma-style bespoke reasoning channel -- no standard markers.
    _GEMMA_TPL = (
        "{% if enable_thinking %}<|think|>{% endif %}<|channel>thought<channel|>"
    )
    # always-on template whose GENERATION PROMPT opens an unclosed <think> (DeepSeek-R1 / QwQ /
    # Qwen3-Thinking shape): the model emits only the closing </think>, so prefill.
    _ALWAYS_ON_OPEN_TPL = (
        "{% for m in messages %}{{ m['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|><think>\n{% endif %}"
    )
    # always-on template that renders PAST assistant <think>...</think> history but leaves the
    # generation prompt open with no <think> (Kimi-K2-Thinking shape): the model self-emits its
    # own block, so prefill mode would blank a normal answer.
    _ALWAYS_ON_HISTORY_TPL = (
        "{% for m in messages %}"
        "{% if m['role'] == 'assistant' %}<think>{{ m.get('reasoning_content', '') }}</think>"
        "{{ m['content'] }}{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_assistant|>assistant<|im_middle|>{% endif %}"
    )

    def _features(self, **over):
        base = {
            "supports_reasoning": True,
            "reasoning_always_on": False,
            "reasoning_style": "enable_thinking",
        }
        base.update(over)
        return base

    def test_g1_enable_thinking_true(self):
        # G1: Qwen3.5 template + explicit enable_thinking=True -> prefilled.
        from routes.inference import _sf_reasoning_prefill_mode
        assert (
            _sf_reasoning_prefill_mode(self._features(), True, self._QWEN_TPL) is True
        )

    def test_g2_enable_thinking_none_defaults_on(self):
        # G2: default request (None) -> prefilled (Qwen3/GLM templates default on).
        from routes.inference import _sf_reasoning_prefill_mode
        assert (
            _sf_reasoning_prefill_mode(self._features(), None, self._QWEN_TPL) is True
        )

    def test_g3_enable_thinking_false(self):
        # G3: thinking explicitly off -> not prefilled.
        from routes.inference import _sf_reasoning_prefill_mode
        assert (
            _sf_reasoning_prefill_mode(self._features(), False, self._QWEN_TPL) is False
        )

    def test_g4_gpt_oss_reasoning_effort_excluded(self):
        # G4: gpt-oss uses explicit tags via HarmonyTextStreamer -> normal mode.
        from routes.inference import _sf_reasoning_prefill_mode
        feats = self._features(reasoning_style = "reasoning_effort")
        assert _sf_reasoning_prefill_mode(feats, True, self._QWEN_TPL) is False

    def test_g5_enable_thinking_effort_included(self):
        # G5: GLM-style enable_thinking_effort also prefills.
        from routes.inference import _sf_reasoning_prefill_mode
        feats = self._features(reasoning_style = "enable_thinking_effort")
        assert _sf_reasoning_prefill_mode(feats, None, self._QWEN_TPL) is True

    def test_g6_non_reasoning_model(self):
        # G6: no reasoning capability -> never prefilled.
        from routes.inference import _sf_reasoning_prefill_mode
        feats = self._features(supports_reasoning = False, reasoning_style = None)
        assert _sf_reasoning_prefill_mode(feats, True, self._QWEN_TPL) is False

    def test_g7_reasoning_always_on_prompt_opens_think(self):
        # G7: always-on template whose generation prompt opens <think> -> prefilled regardless of the flag.
        from routes.inference import _sf_reasoning_prefill_mode
        feats = self._features(reasoning_always_on = True)
        assert (
            _sf_reasoning_prefill_mode(feats, False, self._ALWAYS_ON_OPEN_TPL) is True
        )

    def test_g7b_reasoning_always_on_history_only_not_prefilled(self):
        # G7b (#5704): always-on classification from rendered assistant HISTORY <think></think>
        # (Kimi-K2-Thinking) whose generation prompt opens no <think>. Prefill mode would capture a
        # normal answer entirely as reasoning_content and blank the visible answer, so it must be off.
        from routes.inference import _sf_reasoning_prefill_mode
        feats = self._features(reasoning_always_on = True)
        assert (
            _sf_reasoning_prefill_mode(feats, None, self._ALWAYS_ON_HISTORY_TPL)
            is False
        )

    def test_g8_gemma_bespoke_channel_excluded(self):
        # G8: gemma's <|think|>/<|channel> format has no </think> -> NOT prefilled
        # (would otherwise swallow the whole answer as reasoning). Regression guard.
        from routes.inference import _sf_reasoning_prefill_mode
        assert (
            _sf_reasoning_prefill_mode(self._features(), True, self._GEMMA_TPL) is False
        )

    def test_g9_missing_template_not_prefilled(self):
        # G9: no template available -> conservative (not prefilled).
        from routes.inference import _sf_reasoning_prefill_mode
        assert _sf_reasoning_prefill_mode(self._features(), True, None) is False
