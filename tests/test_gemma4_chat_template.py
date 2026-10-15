import os
import re

import pytest
from jinja2 import Environment, StrictUndefined
from jinja2.exceptions import TemplateError


CHAT_TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "unsloth",
    "chat_templates.py",
)


def _extract_template(name):
    src = open(CHAT_TEMPLATES_PATH, encoding = "utf-8").read()
    pattern = rf'{re.escape(name)}\s*=\s*\\\n"""(.*?)"""'
    m = re.search(pattern, src, flags = re.DOTALL)
    assert m, f"Could not extract {name} from chat_templates.py"
    return m.group(1)


def _env():
    env = Environment(undefined = StrictUndefined, trim_blocks = False, lstrip_blocks = False)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(TemplateError(msg))
    return env


def _render(template_name, messages, **kwargs):
    src = _extract_template(template_name)
    tmpl = _env().from_string(src)
    ctx = {"messages": messages, "add_generation_prompt": False}
    ctx.update(kwargs)
    return tmpl.render(**ctx)


def test_system_message_emits_dedicated_system_turn():
    msgs = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hi"},
    ]
    out = _render("gemma4_template", msgs)
    assert "<|turn>system\nYou are helpful<turn|>" in out
    assert "<|turn>user\nHi<turn|>" in out
    assert "You are helpful\n\nHi" not in out


def test_developer_role_treated_as_system():
    msgs = [
        {"role": "developer", "content": "Internal instructions"},
        {"role": "user", "content": "Hi"},
    ]
    out = _render("gemma4_template", msgs)
    assert "<|turn>system\nInternal instructions<turn|>" in out


def test_no_system_no_thinking_unchanged():
    msgs = [{"role": "user", "content": "Hi"}]
    out = _render("gemma4_template", msgs)
    assert "<|turn>user\nHi<turn|>" in out
    assert "<|turn>system" not in out


def test_assistant_role_renders_as_model_turn():
    msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
    out = _render("gemma4_template", msgs)
    assert "<|turn>model\nA<turn|>" in out
    assert "<|turn>assistant" not in out


def test_thinking_template_defaults_to_thinking_off_when_unset():
    msgs = [{"role": "user", "content": "Hi"}]
    out = _render("gemma4_thinking_template", msgs)
    assert "<|think|>" not in out
    assert "<|turn>system" not in out


def test_thinking_template_emits_think_with_newline_when_enabled():
    msgs = [{"role": "system", "content": "Sys"}, {"role": "user", "content": "Hi"}]
    out = _render("gemma4_thinking_template", msgs, enable_thinking = True)
    assert "<|turn>system\n<|think|>\nSys<turn|>" in out


def test_alternation_violation_raises_template_error():
    msgs = [{"role": "user", "content": "A"}, {"role": "user", "content": "B"}]
    with pytest.raises(TemplateError):
        _render("gemma4_template", msgs)


def test_strip_thinking_strips_matched_pair():
    msgs = [
        {"role": "user", "content": "Q"},
        {
            "role": "assistant",
            "content": "<|channel>thought\n2+2=4<channel|>The answer is 4.",
        },
    ]
    out = _render("gemma4_template", msgs)
    assert "thought" not in out
    assert "2+2=4" not in out
    assert "The answer is 4." in out


def test_strip_thinking_applied_unconditionally_to_model_turn():
    msgs = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "<|channel>reasoning<channel|>final"},
    ]
    for agp in (True, False):
        out = _render("gemma4_template", msgs, add_generation_prompt = agp)
        assert "reasoning" not in out
        assert "final" in out


def test_strip_thinking_applies_to_iterable_text():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "Q"}]},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|channel>r<channel|>final"}],
        },
    ]
    out = _render("gemma4_thinking_template", msgs)
    assert "final" in out
    assert "<|channel>" not in out


def test_strip_thinking_preserves_plain_text():
    msgs = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "plain answer with no markup"},
    ]
    out = _render("gemma4_template", msgs, add_generation_prompt = True)
    assert "plain answer with no markup" in out


def test_multi_turn_strips_all_historical_model_turns():
    msgs = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "<|channel>r1<channel|>A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "<|channel>r2<channel|>A2"},
    ]
    out = _render("gemma4_thinking_template", msgs, add_generation_prompt = True)
    assert "r1" not in out and "r2" not in out
    assert "A1" in out and "A2" in out


def test_thinking_template_injects_empty_thought_channel_by_default():
    msgs = [{"role": "user", "content": "Hi"}]
    out = _render("gemma4_thinking_template", msgs, add_generation_prompt = True)
    assert out.endswith("<|turn>model\n<|channel>thought\n<channel|>")


def test_thinking_template_no_injection_when_thinking_enabled():
    msgs = [{"role": "user", "content": "Hi"}]
    out = _render(
        "gemma4_thinking_template",
        msgs,
        add_generation_prompt = True,
        enable_thinking = True,
    )
    assert "<|channel>thought" not in out


def test_base_template_has_no_channel_thought_injection():
    msgs = [{"role": "user", "content": "Hi"}]
    out = _render("gemma4_template", msgs, add_generation_prompt = True)
    assert out.endswith("<|turn>model\n")
    assert "<|channel>thought" not in out


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }
]
TOOL_CONVO = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Paris"}},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    {"role": "assistant", "content": "It is sunny."},
]
TOOL_SYSTEM = (
    '<|turn>system\n<|tool>declaration:get_weather{description:<|"|>Get the weather<|"|>,'
    'parameters:{properties:{city:{description:<|"|>City name<|"|>,type:<|"|>STRING<|"|>}},'
    'required:[<|"|>city<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>\n'
)
TOOL_ROUND = (
    "<|turn>user\nWeather in Paris?<turn|>\n<|turn>model\n"
    '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
    '<|tool_response>response:get_weather{value:<|"|>sunny<|"|>}<tool_response|>'
)


def _render_hf(template_name, messages, **kwargs):
    env = Environment(trim_blocks = True, lstrip_blocks = True)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(TemplateError(msg))
    ctx = {"messages": messages, "add_generation_prompt": False, "tools": None}
    ctx.update(kwargs)
    return env.from_string(_extract_template(template_name)).render(**ctx)


@pytest.mark.parametrize("name", ["gemma4_template", "gemma4_thinking_template"])
def test_tool_calling_conversation_renders_like_gemma4(name):
    out = _render_hf(name, TOOL_CONVO, tools = TOOLS)
    assert out == TOOL_SYSTEM + TOOL_ROUND + "It is sunny.<turn|>\n"


@pytest.mark.parametrize("name", ["gemma4_template", "gemma4_thinking_template"])
def test_generation_after_tool_result_continues_the_model_turn(name):
    out = _render_hf(name, TOOL_CONVO[:3], tools = TOOLS, add_generation_prompt = True)
    assert out == TOOL_SYSTEM + TOOL_ROUND
