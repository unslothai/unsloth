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
    src = open(CHAT_TEMPLATES_PATH).read()
    pattern = rf'{re.escape(name)}\s*=\s*\\\n"""(.*?)"""'
    m = re.search(pattern, src, flags = re.DOTALL)
    assert m, f"Could not extract {name} from chat_templates.py"
    return m.group(1)


def _env():
    env = Environment(undefined = StrictUndefined, trim_blocks = False, lstrip_blocks = False)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(
        TemplateError(msg)
    )
    return env


def _render(template_name, messages, **kwargs):
    src = _extract_template(template_name)
    tmpl = _env().from_string(src)
    ctx = {"messages": messages, "add_generation_prompt": False}
    ctx.update(kwargs)
    return tmpl.render(**ctx)


# ---------- system turn and <|think|> placement ----------


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


# ---------- strip_thinking macro semantics ----------


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


# ---------- thinking-template gen-prompt injection ----------


def test_thinking_template_injects_empty_thought_channel_by_default():
    # Author defaults enable_thinking=False, so the gen-prompt injection fires.
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
