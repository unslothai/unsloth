# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise the intermediate adapters between request isolation and tool launch."""

import importlib
import sys
import threading

import pytest

from core.inference import os_sandbox


@pytest.mark.parametrize("mode", ["auto", "required"])
@pytest.mark.parametrize(
    "module_name,class_name",
    [
        ("orchestrator", "InferenceOrchestrator"),
        ("inference", "InferenceBackend"),
    ],
)
def test_safetensors_wrappers_preserve_isolation(
    monkeypatch, tmp_path, mode, module_name, class_name
):
    from core.inference import safetensors_agentic, chat_template_helpers

    cls = getattr(importlib.import_module("core.inference." + module_name), class_name)
    backend = cls.__new__(cls)
    backend.active_model_name = "review-model"
    backend.models = {"review-model": {"context_length": 4096}}
    monkeypatch.setattr(chat_template_helpers, "mapped_chat_template", lambda *a, **k: None)
    monkeypatch.setattr(chat_template_helpers, "markup_for_tokenizer", lambda *a, **k: None)
    monkeypatch.setattr(chat_template_helpers, "renderable_tool_catalog", lambda *a, **k: [])
    seen = {}

    def capture_loop(**kwargs):
        seen.update(kwargs)
        return iter(())

    monkeypatch.setattr(safetensors_agentic, "run_safetensors_tool_loop", capture_loop)
    list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "print a marker"}],
            tools = [{"type": "function", "function": {"name": "python"}}],
            tool_execution_mode = mode,
        )
    )
    assert seen["tool_execution_mode"] == mode
    assert_unavailable_launch(monkeypatch, tmp_path, seen["tool_execution_mode"])


def assert_unavailable_launch(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(
        os_sandbox,
        "capability_snapshot",
        lambda **kwargs: os_sandbox.SandboxCapability("none", False, "unavailable test host"),
    )
    starts = []
    monkeypatch.setattr(os_sandbox.subprocess, "Popen", lambda *a, **k: starts.append(a))
    plan = os_sandbox.ToolLaunchPlan(
        (sys.executable, "-c", "print(1)"),
        str(tmp_path),
        {},
        requested_mode = mode,
    )
    if mode == "required":
        with pytest.raises(os_sandbox.SandboxUnavailableError):
            os_sandbox.spawn_prepared_launch(os_sandbox.prepare_tool_launch(plan))
        assert starts == []
    else:
        prepared = os_sandbox.prepare_tool_launch(plan)
        os_sandbox.spawn_prepared_launch(prepared)
        assert len(starts) == 1
        assert not prepared.execution_record.os_isolation


@pytest.mark.parametrize("mode", [None, "auto", "required"])
def test_codex_policy_reaches_shared_launch_policy(monkeypatch, tmp_path, mode):
    from core.inference import openai_codex_tool_loop as codex

    captured = {}

    def shared_loop(transport, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(codex, "stream_with_studio_tools", shared_loop)
    policy = codex.CodexToolPolicy(
        tools = [],
        max_calls = 2,
        timeout = 10,
        permission_mode = "auto",
        confirm_calls = False,
        bypass_permissions = False,
        rag_scope = None,
        **({"tool_execution_mode": mode} if mode is not None else {}),
    )
    codex.stream_codex_with_studio_tools(
        object(),
        run = codex.CodexRunContext("provider", "thread", "session", [], "model", None),
        policy = policy,
        cancel_event = threading.Event(),
    )
    assert captured["policy"].tool_execution_mode == (mode or "auto")
    assert_unavailable_launch(monkeypatch, tmp_path, captured["policy"].tool_execution_mode)
