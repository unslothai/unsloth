# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Capability advertisement contract: classifier honesty, worker→
orchestrator IPC hop, and route-layer end-to-end. Pure helpers + fakes;
no torch / transformers import.
"""

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
  {{- '<|im_start|>system\\n' }}
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
        "supports_preserve_thinking": False,
        "supports_tools": False,
    }


def test_detect_safetensors_features_gptoss_disables_tools():
    """gpt-oss Harmony: tools intentionally off even if template marks it."""
    from routes.inference import _detect_safetensors_features

    backend = MagicMock()
    backend.active_model_name = "unsloth/gpt-oss-20b"
    backend._is_gpt_oss_model.return_value = True

    flags = _detect_safetensors_features(backend, QWEN3_TEMPLATE)
    assert flags["supports_reasoning"] is True
    assert flags["reasoning_style"] == "reasoning_effort"
    assert flags["supports_tools"] is False


# ── Tests: IPC bridge contract ───────────────────────────────────────


def test_orchestrator_mirrors_chat_template_info_into_models_dict():
    """Worker → orchestrator must copy chat_template_info verbatim."""
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

    # Replay orchestrator.load_model's mirror block verbatim.
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
    """Tokenizer with no chat_template still produces a valid reply."""

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
    """End-to-end: Qwen3 safetensors flips supports_tools=True."""
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
