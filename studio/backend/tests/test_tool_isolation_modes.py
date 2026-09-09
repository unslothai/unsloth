# SPDX-License-Identifier: AGPL-3.0-only
import sys

import pytest


@pytest.mark.parametrize(
    "model", ["ChatCompletionRequest", "ResponsesRequest", "AnthropicMessagesRequest"]
)
@pytest.mark.parametrize(
    "platform,default", [("win32", "auto"), ("linux", "auto"), ("darwin", "auto")]
)
def test_api_modes(monkeypatch, model, platform, default):
    from models import inference
    from pydantic import ValidationError

    monkeypatch.setattr(sys, "platform", platform)

    cls = getattr(inference, model)
    data = {"model": "test", "messages": [], "input": "test", "max_tokens": 1}
    assert cls(**data).tool_execution_mode == default
    assert cls(**data, tool_execution_mode = "auto").tool_execution_mode == "auto"
    assert cls(**data, tool_execution_mode = "required").tool_execution_mode == "required"
    for obsolete in ("limited", "container_isolation", "full", "os_isolation_required"):
        with pytest.raises(ValidationError, match = "Choose tool_execution_mode"):
            cls(**data, tool_execution_mode = obsolete)


@pytest.mark.parametrize(
    "name,description",
    [
        ("ChatCompletionRequest", "OpenAI-compatible chat completion request."),
        ("ResponsesRequest", "OpenAI Responses API request."),
    ],
)
def test_request_schema_preserves_public_description(name, description):
    from models import inference
    schema = getattr(inference, name).model_json_schema()
    assert schema["description"].startswith(description)


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_platform_default_reaches_unavailable_launch(monkeypatch, tmp_path, platform, kind):
    from models import inference
    from core.inference import os_sandbox

    monkeypatch.setattr(sys, "platform", platform)
    request = inference.ChatCompletionRequest(model = "test", messages = [])
    monkeypatch.setattr(
        os_sandbox,
        "capability_snapshot",
        lambda **kwargs: os_sandbox.SandboxCapability("none", False, "test unavailable"),
    )
    starts = []
    monkeypatch.setattr(os_sandbox.subprocess, "Popen", lambda *a, **k: starts.append(1))
    plan = os_sandbox.ToolLaunchPlan(
        ("python",),
        str(tmp_path),
        {},
        requested_mode = request.tool_execution_mode,
        execution_kind = kind,
    )
    os_sandbox.spawn_prepared_launch(os_sandbox.prepare_tool_launch(plan))
    assert starts == [1]
