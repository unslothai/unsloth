import ast
import re
import types
from pathlib import Path

import pytest


def _load_change_system_message():
    # Extract just _change_system_message from chat_templates.py so the test runs
    # without importing unsloth (which needs unsloth_zoo / a GPU). Same pattern as
    # tests/saving/test_is_gpt_oss_detection.py.
    source = Path(__file__).parents[2] / "unsloth" / "chat_templates.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    funcs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_change_system_message"
    ]
    namespace = {
        "re": re,
        "logger": types.SimpleNamespace(warning_once = lambda *a, **k: None),
        "DEFAULT_SYSTEM_MESSAGE": {"unsloth": "You are a helpful assistant to the user"},
    }
    module = ast.Module(body = funcs, type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["_change_system_message"]


CUSTOM = "mycustom"  # not in DEFAULT_SYSTEM_MESSAGE -> no predefined default


def test_custom_template_fills_placeholder():
    # A custom template with a {system_message} placeholder must be filled, not
    # left with the literal placeholder.
    fn = _load_change_system_message()
    template, used = fn("System: {system_message}\nUser:", CUSTOM, "You are a pirate")
    assert template == "System: You are a pirate\nUser:"
    assert "{system_message}" not in template
    assert used == "You are a pirate"


def test_custom_template_preserves_backslashes():
    # Why str.replace and not re.sub: a system message with backslashes (Windows
    # paths, LaTeX, group-like text) must be inserted verbatim. re.sub treats the
    # replacement specially -- r"C:\Users" raises bad-escape, r"\1" is a group ref.
    fn = _load_change_system_message()
    for msg in (r"C:\Users\me", r"\frac{a}{b}", r"see \1 here"):
        template, used = fn("System: {system_message}", CUSTOM, msg)
        assert template == f"System: {msg}"
        assert used == msg


def test_custom_template_requires_system_message():
    # A custom template with a placeholder but no system message must raise,
    # rather than silently leaving the placeholder in.
    fn = _load_change_system_message()
    with pytest.raises(ValueError):
        fn("System: {system_message}", CUSTOM, None)


def test_custom_template_without_placeholder_unchanged():
    fn = _load_change_system_message()
    template, used = fn("System: fixed", CUSTOM, "ignored")
    assert template == "System: fixed"


def test_predefined_template_uses_default_then_override():
    # Predefined templates with a default are unaffected by the change.
    fn = _load_change_system_message()
    t1, u1 = fn("System: {system_message}", "unsloth", None)
    assert t1 == "System: You are a helpful assistant to the user"
    t2, u2 = fn("System: {system_message}", "unsloth", "Custom override")
    assert t2 == "System: Custom override"
    assert u2 == "Custom override"
