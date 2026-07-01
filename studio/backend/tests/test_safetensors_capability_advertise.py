# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Capability advertisement contract: classifier honesty, worker->orchestrator
IPC hop, route-layer end-to-end. Pure helpers + fakes; no torch/transformers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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


# Llama-3 / Mistral advertise tools but emit <|python_tag|> / [TOOL_CALLS],
# which our parser can't read. The route helper must not flip supports_tools=True
# for them, else the UI enables a pill the agentic loop can't honour.

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


def test_detect_safetensors_features_llama3_template_suppresses_tools():
    """Llama-3 emits <|python_tag|>; safetensors loop cannot parse it."""
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/Llama-3.2-3B-Instruct")
    flags = _detect_safetensors_features(backend, LLAMA3_TEMPLATE)
    assert flags["supports_tools"] is False


def test_detect_safetensors_features_mistral_template_keeps_tools_on():
    """Mistral emits [TOOL_CALLS]name{json}, which the safetensors loop now parses
    (the shared bracket-tag parser). The gate must no longer suppress it, or the
    PR's Mistral tool support is unreachable through normal capability detection."""
    from routes.inference import _detect_safetensors_features

    backend = SimpleNamespace(active_model_name = "unsloth/mistral-7b-instruct-v0.3")
    flags = _detect_safetensors_features(backend, MISTRAL_TEMPLATE)
    assert flags["supports_tools"] is True


def test_detect_safetensors_features_qwen_tool_call_keeps_tools_on():
    """Sanity check: gate only suppresses non-Qwen formats."""
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


# =====================================================================
# _sf_reasoning_prefill_mode -- gates the prefilled-<think> extractor so
# safetensors/MLX reach GGUF reasoning-block parity for enable_thinking models.
# =====================================================================


class TestSafetensorsReasoningPrefillGate:
    # A minimal Qwen3-style template with the standard <think>/</think> markers.
    _QWEN_TPL = "{% if enable_thinking %}<think>{% endif %}...</think>..."
    # gemma-style bespoke reasoning channel -- no standard markers.
    _GEMMA_TPL = "{% if enable_thinking %}<|think|>{% endif %}<|channel>thought<channel|>"

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
        assert _sf_reasoning_prefill_mode(self._features(), True, self._QWEN_TPL) is True

    def test_g2_enable_thinking_none_defaults_on(self):
        # G2: default request (None) -> prefilled (Qwen3/GLM templates default on).
        from routes.inference import _sf_reasoning_prefill_mode
        assert _sf_reasoning_prefill_mode(self._features(), None, self._QWEN_TPL) is True

    def test_g3_enable_thinking_false(self):
        # G3: thinking explicitly off -> not prefilled.
        from routes.inference import _sf_reasoning_prefill_mode
        assert _sf_reasoning_prefill_mode(self._features(), False, self._QWEN_TPL) is False

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

    def test_g7_reasoning_always_on(self):
        # G7: hardcoded-<think> template -> prefilled regardless of the flag.
        from routes.inference import _sf_reasoning_prefill_mode
        feats = self._features(reasoning_always_on = True)
        assert _sf_reasoning_prefill_mode(feats, False, self._QWEN_TPL) is True

    def test_g8_gemma_bespoke_channel_excluded(self):
        # G8: gemma's <|think|>/<|channel> format has no </think> -> NOT prefilled
        # (would otherwise swallow the whole answer as reasoning). Regression guard.
        from routes.inference import _sf_reasoning_prefill_mode
        assert _sf_reasoning_prefill_mode(self._features(), True, self._GEMMA_TPL) is False

    def test_g9_missing_template_not_prefilled(self):
        # G9: no template available -> conservative (not prefilled).
        from routes.inference import _sf_reasoning_prefill_mode
        assert _sf_reasoning_prefill_mode(self._features(), True, None) is False
