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
