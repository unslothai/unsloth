# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wiring guard for the plan-without-action ``nudge_tool_calls`` policy.

The request flag is explicit at every boundary. ``None`` follows the shared
process default from ``passthrough_healing.nudge_enabled`` (off unless
``UNSLOTH_TOOL_CALL_NUDGE=1``), while Unsloth may opt in by sending ``True``.

Mechanism (verified here without loading a model):

  * the GGUF loop and external Unsloth loop use the same normalizer;
  * the external route forwards the request flag into ``ToolLoopPolicy``;
  * the API request models default the flag to ``None`` (opt-in / off);
  * the Unsloth-facing routes forward the request's flag, and the Unsloth frontend
    sends ``nudge_tool_calls: true`` -- exercised behaviourally in
    ``test_safetensors_tool_loop.py`` and ``test_llama_cpp_tool_loop.py``.
"""

import inspect
import pathlib

from core.inference.llama_cpp import LlamaCppBackend
from core.inference.orchestrator import InferenceOrchestrator
from core.inference.passthrough_healing import nudge_enabled
from core.inference.safetensors_agentic import run_safetensors_tool_loop
from core.inference.studio_tool_loop import ToolLoopPolicy, stream_with_studio_tools


_CHAT_ADAPTER_SOURCE = (
    pathlib.Path(__file__).resolve().parents[2]
    / "frontend"
    / "src"
    / "features"
    / "chat"
    / "api"
    / "chat-adapter.ts"
)

try:
    # core.inference.inference imports unsloth at module scope, which requires
    # unsloth_zoo. The dependency-light backend CI matrix job does not install
    # it, so the safetensors InferenceBackend is folded into the checks below
    # only when the unsloth stack is importable (local runs / full CI); the
    # other entry points are always checked.
    from core.inference.inference import InferenceBackend
except ImportError:
    InferenceBackend = None


def _params(fn):
    return inspect.signature(fn).parameters


def test_shared_loop_accepts_nudge_flag():
    assert "nudge_tool_calls" in _params(run_safetensors_tool_loop)


def test_backends_accept_the_flag():
    methods = [
        InferenceOrchestrator.generate_chat_completion_with_tools,
        LlamaCppBackend.generate_chat_completion_with_tools,
    ]
    if InferenceBackend is not None:  # safetensors path; needs the unsloth stack
        methods.append(InferenceBackend.generate_chat_completion_with_tools)
    for method in methods:
        assert "nudge_tool_calls" in _params(method), method.__qualname__


def test_delegating_backends_forward_the_flag_to_the_shared_loop():
    # safetensors (in-process transformers) and MLX (parent-process orchestrator)
    # both delegate to run_safetensors_tool_loop; GGUF runs its own in-file loop
    # and consumes the flag directly (asserted separately by the gate test).
    methods = [InferenceOrchestrator.generate_chat_completion_with_tools]
    if InferenceBackend is not None:  # safetensors path; needs the unsloth stack
        methods.append(InferenceBackend.generate_chat_completion_with_tools)
    for method in methods:
        src = inspect.getsource(method)
        assert "nudge_tool_calls = nudge_tool_calls" in src, method.__qualname__


def test_gguf_and_external_loops_use_the_shared_nudge_normalizer():
    gguf_src = inspect.getsource(LlamaCppBackend.generate_chat_completion_with_tools)
    assert "_nudge_enabled(nudge_tool_calls)" in gguf_src
    external_src = inspect.getsource(stream_with_studio_tools)
    assert "nudge_enabled(policy.nudge_tool_calls)" in external_src
    assert "nudge_tool_calls" in ToolLoopPolicy.__dataclass_fields__


def test_nudge_normalizer_uses_the_process_default_and_explicit_values(monkeypatch):
    from core.inference import passthrough_healing

    monkeypatch.setattr(passthrough_healing, "_NUDGE_DEFAULT", False)
    assert nudge_enabled(None) is False
    assert nudge_enabled(False) is False
    assert nudge_enabled(True) is True

    monkeypatch.setattr(passthrough_healing, "_NUDGE_DEFAULT", True)
    assert nudge_enabled(None) is True


def test_api_request_models_default_the_flag_off():
    from models.inference import AnthropicMessagesRequest, ChatCompletionRequest
    for model in (ChatCompletionRequest, AnthropicMessagesRequest):
        field = model.model_fields["nudge_tool_calls"]
        assert field.default is None, model.__name__


def test_studio_routes_forward_the_request_flag():
    # The Unsloth chat frontend posts to /v1/chat/completions and /v1/messages
    # with nudge_tool_calls=true; the route handlers forward the request value
    # (external API clients that omit it fall back to the opt-in default).
    from routes import inference as routes_inference
    for handler in (
        routes_inference.produce_openai_chat_completions,
        routes_inference.anthropic_messages,
    ):
        src = inspect.getsource(handler)
        assert "nudge_tool_calls = payload.nudge_tool_calls" in src, handler.__name__


def test_studio_external_adapter_forwards_the_nudge_flag():
    src = _CHAT_ADAPTER_SOURCE.read_text(encoding = "utf-8")
    assert "nudge_tool_calls: runtime.nudgeToolCalls" in src
