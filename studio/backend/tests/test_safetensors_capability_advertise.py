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


# The shipped DeepSeek-V4-Flash template only branches on reasoning_effort 'max', so a literal
# scan surfaces only ['max']; the classifier adds 'high' to expose the full none/high/max ladder.
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




def test_detect_reasoning_flags_qwen3_supports_tools_and_reasoning():
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(QWEN3_TEMPLATE, "unsloth/Qwen3-0.6B")
    assert flags["supports_tools"] is True
    assert flags["supports_reasoning"] is True
    assert flags["reasoning_style"] == "enable_thinking"
    assert flags["supports_preserve_thinking"] is True
    assert flags["preserve_thinking_default"] is False
    assert flags["reasoning_always_on"] is False


def test_detect_reasoning_flags_qwen38_defaults_preserve_thinking_on():
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(QWEN3_TEMPLATE, "unsloth/Qwen3.8-27B-GGUF")
    assert flags["supports_preserve_thinking"] is True
    assert flags["preserve_thinking_default"] is True


@pytest.mark.parametrize(
    "model_id",
    [
        "unsloth/Qwen3.6-27B-GGUF",
        "unsloth/Qwen3.80-27B-GGUF",
        "custom/myQwen3.8-27B",
    ],
)
def test_preserve_thinking_default_does_not_leak_to_other_model_families(model_id):
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(QWEN3_TEMPLATE, model_id)
    assert flags["supports_preserve_thinking"] is True
    assert flags["preserve_thinking_default"] is False


def test_qwen38_without_the_kwarg_does_not_invent_preserve_support():
    from core.inference.llama_cpp import detect_reasoning_flags

    flags = detect_reasoning_flags(PLAIN_TEMPLATE, "custom/Qwen3.8-27B")
    assert flags["supports_preserve_thinking"] is False
    assert flags["preserve_thinking_default"] is False


def test_qwen38_default_keeps_prior_reasoning_in_the_template():
    pytest.importorskip("jinja2")
    from jinja2 import BaseLoader, Environment
    from core.inference.llama_cpp import detect_reasoning_flags

    render = Environment(loader = BaseLoader()).from_string(QWEN3_TEMPLATE)
    common = {
        "tools": [],
        "messages": [],
        "enable_thinking": True,
        "assistant": {"reasoning_content": "SECRET_THOUGHT"},
    }
    before = detect_reasoning_flags(QWEN3_TEMPLATE, "unsloth/Qwen3.6-27B-GGUF")
    after = detect_reasoning_flags(QWEN3_TEMPLATE, "unsloth/Qwen3.8-27B-GGUF")

    assert "SECRET_THOUGHT" not in render.render(
        **common,
        preserve_thinking = before["preserve_thinking_default"],
    )
    assert "SECRET_THOUGHT" in render.render(
        **common,
        preserve_thinking = after["preserve_thinking_default"],
    )


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

    flags = detect_reasoning_flags(DEEPSEEK_V4_TEMPLATE, "unsloth/DeepSeek-V4-Flash-GGUF")
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
        "preserve_thinking_default": False,
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


# Llama-3 / Mistral / Gemma 4 formats are parser-supported; only templates matching none are suppressed.

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


# DeepSeek V3 / V3.1 / R1 blocks use the full-width pipe (U+FF5C) and lower-1/8-block (U+2581).
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


# Kimi K2 / Moonshot uses ``functions.NAME:IDX`` as the per-call id.
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
    from routes.inference import _detect_safetensors_features, _sf_reasoning_prefill_mode

    tpl_with_gemma_native = "{% if add_generation_prompt %}<|channel>thought\n<channel|>{% endif %}"
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
                "chat_template_info": {"template": "{% if tools %}<tool_call>{% endif %}"},
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


# Qwen3.5 family pin: a faithful slice of the live templates so the classifier cannot silently regress.

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
    _entry = _bm.get(mc.identifier) or _bm.get(getattr(backend, "active_model_name", None)) or {}
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
            self.models = {"legacy/no-template": {}}

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


def test_route_layer_emits_preserve_default_true_for_qwen38_safetensors():
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(
        active_model_name = "unsloth/Qwen3.8-27B",
        models = {},
    )

    flags = _detect_safetensors_features(backend, QWEN3_TEMPLATE)

    assert flags["supports_preserve_thinking"] is True
    assert flags["preserve_thinking_default"] is True


@pytest.mark.parametrize(
    "opener",
    [
        "<｜tool▁calls▁begin｜>",
        "<｜tool_calls_begin｜>",
        "<｜tool▁calls｜>",
        "<｜tool calls begin｜>",
        "<｜tool\\_calls\\_begin｜>",
    ],
)
def test_detect_safetensors_features_deepseek_opener_variants_keep_tools_on(opener):
    # The route gate derives its markers from the parser's TOOL_XML_SIGNALS so it cannot drift behind the parser.
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


# Templates that advertise tools and prompt the bare-JSON form, but whose example is pretty-printed or JSON-escaped.
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
    # The parser accepts that whitespace via raw_decode, so supports_tools stays on.
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
    # Negative control: tools advertised but no recognised emission form, so the gate matches nothing extra.
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/Llama-3.2-3B-Instruct")
    flags = _detect_safetensors_features(backend, _TOOLS_ADVERTISED_NO_PARSEABLE_FORM)
    assert flags["supports_tools"] is False


def test_detect_safetensors_features_keeps_tools_for_function_alias_bare_json():
    # The parser-supported {"function":...} alias must keep supports_tools, mirroring {"name":...}.
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
    _QWEN35_TPL = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n"
        "{% if enable_thinking is defined and enable_thinking is true %}<think>\n"
        "{% else %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
    )
    _QWEN3_TPL = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n"
        "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
        "{% endif %}"
    )
    _GEMMA_TPL = "{% if enable_thinking %}<|think|>{% endif %}<|channel>thought<channel|>"
    _PROMPT_OPENS_THINK_TPL = (
        "{% for m in messages %}{{ m['content'] }}{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|><think>\n{% endif %}"
    )
    _HISTORY_ONLY_THINK_TPL = (
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
        from routes.inference import _sf_reasoning_prefill_mode
        assert _sf_reasoning_prefill_mode(self._features(), True, self._QWEN35_TPL) is True

    def test_g2_enable_thinking_none_follows_template_default(self):
        # With the kwarg omitted the template's default decides; reading it as prefilled captured the answer as thought.
        from routes.inference import _sf_reasoning_prefill_mode
        assert _sf_reasoning_prefill_mode(self._features(), None, self._QWEN35_TPL) is False

    def test_g2b_self_emitting_template_not_prefilled(self):
        # Thinking is on but the prompt opens no <think>, so the extractor starts normal.
        from routes.inference import _sf_reasoning_prefill_mode
        assert _sf_reasoning_prefill_mode(self._features(), None, self._QWEN3_TPL) is False
        assert _sf_reasoning_prefill_mode(self._features(), True, self._QWEN3_TPL) is False

    def test_g3_enable_thinking_false(self):
        from routes.inference import _sf_reasoning_prefill_mode
        assert _sf_reasoning_prefill_mode(self._features(), False, self._QWEN35_TPL) is False

    def test_g4_gpt_oss_reasoning_effort_excluded(self):
        # gpt-oss uses explicit tags via HarmonyTextStreamer, so normal mode.
        from routes.inference import _sf_reasoning_prefill_mode
        feats = self._features(reasoning_style = "reasoning_effort")
        assert _sf_reasoning_prefill_mode(feats, True, self._PROMPT_OPENS_THINK_TPL) is False

    def test_g5_enable_thinking_effort_included(self):
        # enable_thinking_effort is not excluded by the style gate.
        from routes.inference import _sf_reasoning_prefill_mode
        feats = self._features(reasoning_style = "enable_thinking_effort")
        assert _sf_reasoning_prefill_mode(feats, None, self._PROMPT_OPENS_THINK_TPL) is True

    def test_g6_non_reasoning_model(self):
        from routes.inference import _sf_reasoning_prefill_mode
        feats = self._features(supports_reasoning = False, reasoning_style = None)
        assert _sf_reasoning_prefill_mode(feats, True, self._PROMPT_OPENS_THINK_TPL) is False

    def test_g7_reasoning_always_on_prompt_opens_think(self):
        # An always-on template whose generation prompt opens <think> is prefilled regardless of the flag.
        from routes.inference import _sf_reasoning_prefill_mode
        feats = self._features(reasoning_always_on = True)
        assert _sf_reasoning_prefill_mode(feats, False, self._PROMPT_OPENS_THINK_TPL) is True

    def test_g7b_reasoning_always_on_history_only_not_prefilled(self):
        # #5704: always-on classification from rendered assistant HISTORY (Kimi-K2-Thinking) whose
        # generation prompt opens no <think>. Prefill mode would capture a normal answer as reasoning.
        from routes.inference import _sf_reasoning_prefill_mode
        feats = self._features(reasoning_always_on = True)
        assert _sf_reasoning_prefill_mode(feats, None, self._HISTORY_ONLY_THINK_TPL) is False

    def test_g8_gemma_bespoke_channel_excluded(self):
        # gemma's <
        from routes.inference import _sf_reasoning_prefill_mode
        assert _sf_reasoning_prefill_mode(self._features(), True, self._GEMMA_TPL) is False

    def test_g9_missing_template_not_prefilled(self):
        # No template available -> conservative (not prefilled).
        from routes.inference import _sf_reasoning_prefill_mode
        assert _sf_reasoning_prefill_mode(self._features(), True, None) is False
