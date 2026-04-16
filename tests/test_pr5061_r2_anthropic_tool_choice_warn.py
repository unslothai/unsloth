import os, sys
from unittest.mock import patch, MagicMock

_backend = os.path.join(os.path.dirname(__file__), "..", "studio", "backend")
sys.path.insert(0, _backend)

from routes import inference as inf_mod


def _simulate_coerce(tool_choice):
    """Mimic the exact snippet that lives at the top of anthropic_messages()."""
    translated = inf_mod.anthropic_tool_choice_to_openai(tool_choice)
    if translated is None:
        if tool_choice is not None:
            inf_mod.logger.warning(
                "anthropic_messages.tool_choice_unrecognized",
                tool_choice = tool_choice,
            )
        translated = "auto"
    return translated


def test_unrecognized_dict_warns_and_falls_back_to_auto():
    with patch.object(inf_mod.logger, "warning") as mock_warn:
        result = _simulate_coerce({"type": "wibble"})
    assert result == "auto"
    mock_warn.assert_called_once()
    assert mock_warn.call_args.args[0] == "anthropic_messages.tool_choice_unrecognized"


def test_non_dict_input_warns_and_falls_back():
    with patch.object(inf_mod.logger, "warning") as mock_warn:
        result = _simulate_coerce("auto_string")
    assert result == "auto"
    mock_warn.assert_called_once()


def test_none_input_no_warning():
    with patch.object(inf_mod.logger, "warning") as mock_warn:
        result = _simulate_coerce(None)
    assert result == "auto"
    mock_warn.assert_not_called()


def test_recognized_input_no_warning():
    with patch.object(inf_mod.logger, "warning") as mock_warn:
        result = _simulate_coerce({"type": "any"})
    assert result == "required"
    mock_warn.assert_not_called()
