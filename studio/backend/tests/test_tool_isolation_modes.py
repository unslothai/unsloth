# SPDX-License-Identifier: AGPL-3.0-only
import pytest


@pytest.mark.parametrize(
    "model", ["ChatCompletionRequest", "ResponsesRequest", "AnthropicMessagesRequest"]
)
def test_api_modes(model):
    from models import inference
    from pydantic import ValidationError

    cls = getattr(inference, model)
    data = {"model": "test", "messages": [], "input": "test", "max_tokens": 1}
    assert cls(**data).tool_execution_mode == "auto"
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
