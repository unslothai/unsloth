# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wiring guard for the plan-without-action ``nudge_tool_calls`` policy.

Decided policy: the re-prompt is ALWAYS ON for the Studio inference paths
(safetensors, GGUF/llama_cpp, MLX) and OPT-IN for the API (/v1 OpenAI-compat +
Anthropic-compat, controlled by the request's ``nudge_tool_calls``, default off).

Mechanism (verified here without loading a model):

  * every backend tool-loop entry point accepts and forwards ``nudge_tool_calls``
    (safetensors -> ``InferenceBackend``; MLX -> ``InferenceOrchestrator``; both
    call the shared ``run_safetensors_tool_loop``; GGUF -> ``LlamaCppBackend``);
  * the safetensors/MLX loop gates the retry on a truthy flag (new retry ->
    opt-in), while the GGUF loop keeps its pre-existing default-on behaviour
    (``None`` keeps nudging) so an omitted flag never disables GGUF;
  * the API request models default the flag to ``None`` (opt-in / off);
  * the Studio-facing routes forward the request's flag, and the Studio frontend
    sends ``nudge_tool_calls: true`` -- exercised behaviourally in
    ``test_safetensors_tool_loop.py`` and ``test_llama_cpp_tool_loop.py``.
"""

import inspect

from core.inference.llama_cpp import LlamaCppBackend
from core.inference.orchestrator import InferenceOrchestrator
from core.inference.safetensors_agentic import run_safetensors_tool_loop

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


def test_safetensors_loop_is_opt_in_while_gguf_stays_default_on():
    # Safetensors/MLX: the retry is new here, so it requires a truthy flag.
    sf_src = inspect.getsource(run_safetensors_tool_loop)
    assert "and nudge_tool_calls" in sf_src
    # GGUF: pre-existing nudge must not be accidentally disabled -- an omitted
    # (None) flag keeps nudging; only an explicit False turns it off.
    gguf_src = inspect.getsource(LlamaCppBackend.generate_chat_completion_with_tools)
    assert "nudge_tool_calls is None or nudge_tool_calls" in gguf_src


def test_api_request_models_default_the_flag_off():
    from models.inference import AnthropicMessagesRequest, ChatCompletionRequest
    for model in (ChatCompletionRequest, AnthropicMessagesRequest):
        field = model.model_fields["nudge_tool_calls"]
        assert field.default is None, model.__name__


def test_studio_routes_forward_the_request_flag():
    # The Studio chat frontend posts to /v1/chat/completions and /v1/messages
    # with nudge_tool_calls=true; the route handlers forward the request value
    # (external API clients that omit it fall back to the opt-in default).
    from routes import inference as routes_inference
    for handler in (
        routes_inference.openai_chat_completions,
        routes_inference.anthropic_messages,
    ):
        src = inspect.getsource(handler)
        assert "nudge_tool_calls = payload.nudge_tool_calls" in src, handler.__name__
